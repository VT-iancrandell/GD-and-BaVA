source("generate corpus.R")

docset1 = c(1, 0, 0, 0, 0)
docset2 = c(0, 1, 0, 0, 0)

corpus = c(.generate.document(50, topics, docset1, vocab, 100), .generate.document(50, topics, docset2, vocab, 100))
word.matrix = .word.matrix(corpus, vocab)

true.beta = rbind(docset1 %*% topics, docset2 %*% topics)

#############################################
# Begin Implementation of BNJ's LDA Algorithm
#############################################
#Constants
#V: vocab length
#M: corpus length
#N_m: words in mth doc
#k: number of topics
###################
# Latent variables#
###################
# Doc topic mix: theta
# Word topic: z, a draw from multinomial(theta),
###########
#Parameters
###########
#alpha: DP parameter for topic mixture, variationally approximated with gamma
#beta: matrix for probabilities for each word for each topic, data generated using beta = topics, variationally approximated with phi
#k number of topics, considered fixed

bnj.lda = function(word.matrix.list, vocab = NA, k, tol = .001){
  V = length(vocab)
  
  alpha.old = alpha = rep(.5, k)
  beta.old = matrix(runif(V * k), nrow = k, ncol = V)
  beta.old = beta.old / apply(beta.old, 1, sum)
  beta = beta.old
  colnames(beta.old) = colnames(beta) = vocab
  
  N = unlist(lapply(word.matrix.list, nrow))
  M = length(word.matrix.list)
  
  # E step: find p(theta, z | w, alpha, beta). Approximate with q(theta, z | gamma, phi). gamma is variational Dirichlet parameter (n vector, univariate for each doc) and phi is variational Multinomial parameter (n list, entry for each doc, sub entry for each word in each docs)
  
  corpus.phi = vector("list", M)
  corpus.gamma = matrix(0, nrow = M, ncol = k)
  parms.converge = F
  
  while(!parms.converge){
    
    for(m in 1:M){
      var.converge = F
      phi = phi.old = matrix(1/k, nrow = N[m], ncol = k)
      gamma = gamma.old = alpha + N[m]/k
      
      while(!var.converge){
        
        gamma.part = exp(digamma(gamma.old) - digamma(sum(gamma.old)))
        for(n in 1:N[m]){
          phi[n,] = beta %*% word.matrix.list[[m]][n,] * gamma.part
          phi[n,] = phi[n,]/sum(phi[n,])
        }
        gamma = alpha + apply(phi, 2, sum)
        
        # Test for convergence
        phi.change = sum((phi - phi.old)^2) / sum(phi.old^2)
        gamma.change = sum((gamma - gamma.old)^2) / sum(gamma.old^2)
        
        if(phi.change < tol & gamma.change < tol){
          var.converge = T
        }else{
          phi.old = phi
          gamma.old = gamma
        }
      }
      corpus.phi[[m]] = phi
      corpus.gamma[m,] = gamma
    }
    
    # M step, maximixe the variational lower bound found above over alpha and beta
    # This isn't working, so Ill try just optimizing the log likelihood for alpha
#     a.old = alpha.old
#     alpha.converge = F
#     
      #gamma.part = apply(apply(corpus.gamma, 1, function(x){digamma(x) - digamma(sum(x))}), 1, sum)
#     
#     while(!alpha.converge){
#       
#       h = -M*trigamma(a.old)
#       z = M*trigamma(sum(a.old))
#       grad = M * (digamma(sum(a.old)) - digamma(a.old)) + gamma.part
#       c = sum(grad/h)/((1/z) + sum(1/h))
#       hinvg = (grad - c)/h
#       a.new = a.old - hinvg
#       
#       a.change = sum((a.old - a.new)^2)/sum(a.old^2)
#       a.old = a.new
#       if(a.change < tol) alpha.converge = T
#     }
      gamma.seg = apply(corpus.gamma, 1, function(x){digamma(x) - digamma(sum(x))})
      
    alpha.ll = function(alpha, gamma.seg){
      out = 0
      for(m in 1:M){
        out = out + log(gamma(sum(alpha))) - sum(log(gamma(alpha))) + sum((alpha - 1)*gamma.seg[,m])
      }
      return(out)
    }
    a.new = constrOptim(alpha.old, alpha.ll, grad = NULL, gamma.seg = gamma.seg, ui = diag(k), ci = rep(0, k), control = list(fnscale = -1))$par
    
    alpha = a.new
    beta = matrix(0, nrow = k, ncol = V)
    colnames(beta) = vocab
    
    for(d in 1:M){
      tmp.doc = word.matrix.list[[d]]
      tmp.phi = corpus.phi[[d]]
      beta = beta + t(tmp.phi) %*% tmp.doc
    }
    beta = beta / apply(beta, 1, sum)
    
    alpha.change = sum((alpha - alpha.old)^2) / sum(alpha.old^2)
    beta.change = sum((beta - beta.old)^2) / sum(beta.old^2)
    
    if(alpha.change < tol & beta.change < tol){
      parms.converge = T
      print("parameters converged")
    }else{
      alpha.old = alpha
      beta.old = beta
      print("parameters failed to converge")
    }
  }
  return(list(gamma = corpus.gamma, phi = corpus.phi, alpha = alpha, beta = beta))
}

sim.lda = bnj.lda(word.matrix, vocab, 2)

# The modal column for each doc in corpus.gamma is the 'cluster assignment' for that doc. The row is the topic mixture. corpus.phi gives topic probs for each word.

# Cluster documents
doc.clusters = apply(sim.lda$gamma, 1, function(x){which(x == max(x))})

# Count up words for each topic. What words appeared most often in a given topic?

word.clusters = lapply(sim.lda$phi, function(x){
  apply(x ,1, function(y){
    which(y == max(y))
    })
})

category.tables = lapply(tapply(unlist(corpus), unlist(word.clusters), table)
                         ,function(x){
                           sort(x / sum(x), T)
                         })
doc.clusters
category.tables
sim.lda$alpha
sim.lda$beta
sum((true.beta - sim.lda$beta)^2)

