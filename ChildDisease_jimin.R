library(dplyr)
rm(list=ls())
ss = read.csv('/Users/jaylee/Desktop/Data2022/For Students/S5_scores_cleaned.csv')
logs = read.csv('/Users/jaylee/Desktop/Data2022/For Students/logs.csv')
# remove all rows without  player_id
ss <- na.omit(ss) 
a <- logs[,c('player_id','event_id','event_description')] 
an <- a %>% group_by(player_id,event_id,event_description) %>% summarise(event_num=table(event_id))
View(an)
ret <-an %>% group_by(player_id) %>% slice_max(order_by = event_num, n = 5)
View(ret)
tb <- table(ret$event_id)
tb <- as.data.frame(tb)
ret[ret$event_id==c(207,210,211,407,902),]

barplot(tb$Freq~tb$Var1,main="Frequent event",xlab="event_id",ylab="Frequency")


rawdocs = c("Player selects an NPC fact",
            "Player begins to scan Scene using Sense",
            "Player ends scan",
            "Player selects a fact card in the library view",
            "Player panned Scene"
)
set.seed(440)
docs = strsplit(rawdocs, split=" ")
# unique words
vocab = unique(unlist(docs))
# replace words in documents with wordIDs 
for (i in 1:length(docs)) {
docs[[i]] = match( docs[[i]], vocab)}

lda_gibbs = function(docs, 
                     vocab,
                     K,
                     alpha,
                     beta,
                     chain_length)
  
  
{
  V = length(vocab)
  M = length(docs)
  # create word-topic matrix
  wt = matrix(0, K, V)
  colnames(wt) = vocab
  # create topic assignment token list
  ta = lapply(docs, function(x) rep( 0, length(x))) 
  names(ta) = paste0("doc", 1:M)
  # create document-topic matrix
  dt = matrix(0, M, K)
  for(d in 1:M)
  {
    # randomly assign topic to each word in the list
    for( w in 1:length(docs[[d]])) {
      ta[[d]][w] = sample( 1:K, 1 )
      # extract the topic index, word id and update the corresponding cell 
      ti = ta[[d]][w]
      wi = docs[[d]][w]
      wt[ti, wi] = wt[ti, wi] + 1
    }
    # count words in document d assigned to each topic t
    for(t in 1:K)
      dt[d, t] = sum(ta[[d]] == t)
  }
  # MAIN GIBBS SAMPLING LOOP ----------------------------- # for each pass through the corpus
  for (i in 1:chain_length)
  {
    # for each document
    for (d in 1:M) {
      # for each word
      for (w in 1:length(docs[[d]])) {
        t0 = ta[[d]][w]
        wid = docs[[d]][w]
        dt[d, t0] = dt[d, t0] - 1
        wt[t0, wid] = wt[t0, wid] - 1
        post1  = (wt[, wid] + beta) / (rowSums(wt) + V*beta)
        post2 = (dt[d, ] + alpha) / (sum(dt[d, ]) + K *alpha)
        # sample with probability proportional to posterior terms
        t1 = sample(1:K, 1, prob=post1*post2)
        # update topic assignment list with newly sampled topic for token w 
        # and re-increment word-topic and document-topic count matrices with 
        # the new sampled topic for token w.
      
  ta[[d]][w] = t1
  dt[d, t1] = dt[d, t1] + 1
  wt[t1, wid] = wt[t1, wid] + 1
      }}}
  # return renoramlized latent probabilities
  phi = (wt + beta ) / (rowSums(wt) + V * beta)
  theta = (dt + alpha) / (rowSums(dt) + K * alpha)
  return(
    list( wt=wt,
          dt=dt,
          phi=phi,
          theta=theta
    ))
}
lda_res = lda_gibbs(docs, vocab, K=2, alpha=1, beta=.001, chain_length=1000)
rawdocs[order(lda_res$theta[, 1], decreasing=TRUE)]

names(sort(lda_res$phi[1, ], decreasing=TRUE)[1:5])

names(sort(lda_res$phi[2, ], decreasing=TRUE)[1:5])







