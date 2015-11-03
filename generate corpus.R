library(gtools)
vocab = paste(rep(letters, each = 10), rep(0:9, 26), sep = '')

# Topics. Each of these is a simple multinomial probability vector over the vocab. 90% probability is distributed evenly over the appropriate terms, the other 10% over their compliment. This can be modified freely to make your own topics

in.topic.prop = .9

vowels = rep((1 - in.topic.prop) / (length(vocab) - length(grep("[a e i o u]", vocab))), (length(vocab)))
vowels[grep("[a e i o u]", vocab)] = in.topic.prop / length(grep("[a e i o u]", vocab))
consonants = rep((1 - in.topic.prop) / (length(vocab) - length(grep("[b c d f g h j k l m n p q r s t v w x y z]", vocab))), length(vocab))
consonants[grep("[b c d f g h j k l m n p q r s t v w x y z]", vocab)] = in.topic.prop / length(grep("[b c d f g h j k l m n p q r s t v w x y z]", vocab))
primes = rep((1 - in.topic.prop) / (length(vocab) - length(grep("[2 3 5 7]", vocab))), (length(vocab)))
primes[grep("[2 3 5 7]", vocab)] = in.topic.prop / length(grep("[2 3 5 7]", vocab))
odds = rep((1 - in.topic.prop) / (length(vocab) - length(grep("[1 3 5 7 9]", vocab))), (length(vocab)))
odds[grep("[1 3 5 7 9]", vocab)] = in.topic.prop / length(grep("[1 3 5 7 9]", vocab))
evens =  rep((1 - in.topic.prop) / (length(vocab) - length(grep("[0 2 4 6 8]", vocab))), (length(vocab)))
evens[grep("[0 2 4 6 8]", vocab)] = in.topic.prop / length(grep("[0 2 4 6 8]", vocab))

topics = rbind(vowels, consonants, primes, odds, evens)
colnames(topics) = vocab
rm(in.topic.prop, vowels, consonants, primes, odds, evens)

# This will create a set of documents.
# Arguments:
# no.docs: number of documents you'd like
# topics: a matrix of word proportions for a set of topics. nrow = number of topics, ncol = length of the vocab
# topic.props: a vector of proportions for the topic props. length = number of topics
# vocab: the vocabulary you're working with
# doc.lengths: length of the documents in the set. Can be a vector with length = no.docs or a scalar, in which case all docs have the same length

.generate.document = function(no.docs, topics, topic.props = NA, vocab, doc.lengths){
  docs = vector("list", no.docs)
  
  if(anyNA(topic.props)) topic.props = rep(1/dim(topics)[1], dim(topics)[1])
  if(length(doc.lengths) == 1) doc.lengths = rep(doc.lengths, no.docs)
  
  for(i in 1:no.docs){
    for(j in 1:doc.lengths[i]){
      z = sample(1:length(topic.props), 1, prob = topic.props)
      docs[[i]] = c(docs[[i]], sample(vocab, 1, p = topics[z,]))
    }
  }
  return(docs)
}

# This'll make 50 documents and 50 tweets

docset1 = c(.5, 0, 0, .5, 0)
docset2 = c(0, .5, 0, 0, .5)
corpus = c(.generate.document(50, topics, docset1, vocab, 100), .generate.document(50, topics, docset2, vocab, 10))

# Takes a corpus and transforms them into a matrix of word counts. nrow = corpus length, ncol = vocab length

.generate.word.counts = function(corpus, vocab = NA){
  
  if(anyNA(vocab)) vocab = unique(unlist(corpus))
  
  word.counts = matrix(NA, nrow = length(corpus), ncol = length(vocab))
  for(i in 1:length(corpus)){
    for(j in 1:length(vocab)){
      word.counts[i,j] = sum(corpus[[i]] == vocab[j])
    }
  }
  colnames(word.counts) = vocab
  return(word.counts)
}
word.counts = .generate.word.counts(corpus, vocab)

# Your tfidf function

.Raw_TFIDF = function(dtm){
  dtm.mat = dtm
  tf = dtm.mat/apply(dtm.mat,1,sum)
  idf = log(nrow(dtm.mat)/(1+apply(dtm.mat,2,function(c)sum(c!=0))))
  tfidf = dtm.mat
  
  for(word in names(idf))
  {
    tfidf[,word] <- tf[,word] * idf[word]
  }
  
  #tfidfsums=apply(tfidf,1,sum)
  #tfidf=sweep(tfidf,1,tfidfsums,FUN='/') 
  
  return(tfidf)
}

# This expresses text documents as sparse matrices, each row a multinomial vector for the appropriate word a la BNJ. Takes a whole corpus as argument for potential tfidfing later. Can ignore.

.word.matrix = function(corpus, vocab = NA){
  
  if(anyNA(vocab)) vocab = unique(unlist(corpus))
  V = length(vocab)
  
  matrices = lapply(corpus, function(x, V){
    tmp = matrix(0, nrow = length(x), ncol = V)
    for(i in 1:dim(tmp)[1]){
      tmp[i, which(vocab == x[i])] = 1
    }
    return(tmp)
  }, V)
}


# You can ignore this part. It's a version of the LDA code I'm saving while I do something else.

# Storing the old lda function which took the raw docs as input while I implement the version which uses the sparse matrix representation

#bnj.lda = function(corpus, vocab = NA, k, tol = .001){
#   V = length(vocab)
#   
#   alpha.old = alpha = rep(1, k)
#   beta.old = matrix(runif(V * k), nrow = k, ncol = V)
#   beta.old = beta.old / apply(beta.old, 1, sum)
#   beta = beta.old
#   colnames(beta.old) = colnames(beta) = vocab
#   
#   N = unlist(lapply(corpus, length))
#   M = length(corpus)
#   
#   # E step: find p(theta, z | w, alpha, beta). Approximate with q(theta, z | gamma, phi). gamma is variational Dirichlet parameter (n vector, univariate for each doc) and phi is variational Multinomial parameter (n list, entry for each doc, sub entry for each word in each docs)
#   
#   corpus.phi = vector("list", length(corpus))
#   corpus.gamma = matrix(0, nrow = M, ncol = k)
#   parms.converge = F
#   
#   while(!parms.converge){
#     
#     for(m in 1:M){
#       var.converge = F
#       phi = phi.old = matrix(1/k, nrow = N[m], ncol = k)
#       gamma = gamma.old = alpha + N[m]/k
#       
#       while(!var.converge){
#         
#         for(n in 1:N[m]){
#           phi[n,] = beta[, corpus[[m]][n]] * exp(digamma(gamma.old) - digamma(sum(gamma.old)))
#           phi[n,] = phi[n,]/sum(phi[n,])
#         }
#         gamma = alpha + apply(phi, 2, sum)
#         
#         # Test for convergence
#         phi.change = sum((phi - phi.old)^2) / sum(phi.old^2)
#         gamma.change = sum((gamma - gamma.old)^2) / sum(gamma.old^2)
#         
#         if(phi.change < tol & gamma.change < tol){
#           var.converge = T
#           print("phi and gamma converged")
#         }else{
#           phi.old = phi
#           gamma.old = gamma
#         }
#       }
#       corpus.phi[[m]] = phi
#       corpus.gamma[m,] = gamma
#     }
#     
#     # M step, maximixe the variational lower bound found above over alpha and beta
#     
#     a.old = alpha.old
#     alpha.converge = F
#     
#     gamma.part = apply(apply(corpus.gamma, 1, function(x){digamma(x) - digamma(sum(x))}), 1, sum)
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
#     
#     alpha = a.new
#     beta = matrix(0, nrow = k, ncol = V)
#     colnames(beta) = vocab
#     
#     for(d in 1:M){
#       tmp.doc = corpus[[d]]
#       tmp.phi = corpus.phi[[d]]
#       for(n in 1:N[d]){
#         for(i in 1:k){
#           beta[i,which(vocab == tmp.doc[n])] = beta[i,which(vocab == tmp.doc[n])] + tmp.phi[n,i]
#         }
#       }
#     }
#     beta = beta / apply(beta, 1, sum)
#     
#     alpha.change = sum((alpha - alpha.old)^2) / sum(alpha.old^2)
#     beta.change = sum((beta - beta.old)^2) / sum(beta.old^2)
#     
#     if(alpha.change < tol & beta.change < tol){
#       parms.converge = T
#       print("alpha and beta converged")
#     }else{
#       alpha.old = alpha
#       beta.old = beta
#     }
#   }
#   return(list(gamma = corpus.gamma, phi = corpus.phi, alpha = alpha, beta = beta))
# }
#

