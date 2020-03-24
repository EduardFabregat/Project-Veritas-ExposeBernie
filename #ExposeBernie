options(stringsAsFactors = FALSE)

library(quanteda)

bernie$text<-gsub("[h][t][t][p]\\S+","", bernie$text)
bernie$text<-gsub("pic.twitter\\S+","", bernie$text)

bernie$index <- 1:nrow(bernie)

removed_df<-bernie[duplicated(bernie$text),]
bernie2 <- bernie[!duplicated(bernie$text),]

library(cld2)
library(tidyverse)

lan <- detect_language(epstein2$text)
lan <- as.data.frame(lan)
lan_n <- count(lan, lan)

bernie3 <- cbind(bernie2, lan)

bernie_en <- bernie3[bernie3$lan == "en" | bernie3$lan == "NA", ]

bernie_en <- bernie_en[c(-92)]

mycorpus <- corpus(bernie2)

stopwords_and_single<-c(stopwords("english"), "amp", "&amp", LETTERS,letters)
dfm_ber <- dfm(mycorpus, tolower = TRUE, remove_punct = TRUE, 
               remove_numbers=TRUE, remove = stopwords_and_single, 
               stem = FALSE, remove_separators=TRUE)

docnames(dfm_ber) <- dfm_ber@docvars$index

dfm_ber2 <- dfm_trim(dfm_ant, max_docfreq = 0.95, min_docfreq = 0.01, 
                     docfreq_type = "prop")

dtm_lda <- convert(dfm_ber2, to = "topicmodels")

full_data<-dtm_lda

n <- nrow(full_data)

print(Sys.time())
MainresultDF<-data.frame(k=c(1),perplexity=c(1),myalpha=c("x"))
MainresultDF<-MainresultDF[-1,]
candidate_alpha<- c(0.01, 0.05, 0.1, 0.2, 0.5)
candidate_k <- c(seq(1,10)*10, 125)
mycores <- detectCores()

library(doParallel)

for (eachalpha in candidate_alpha) { 
  print ("now running ALPHA:")
  print (eachalpha)
  print(Sys.time())
  #----------------5-fold cross-validation, different numbers of topics----------------
  cluster <- makeCluster(mycores) # leave one CPU spare...
  registerDoParallel(cluster)
  
  clusterEvalQ(cluster, {
    library(topicmodels)
  })
  
  folds <- 5
  splitfolds <- sample(1:folds, n, replace = TRUE)
  #candidate_k <- c(2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100) # candidates for how many topics
  
  #clusterExport(cluster, c("full_data", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k"))
  clusterExport(cluster, c("full_data", "splitfolds", "folds", "candidate_k"))
  
  # we parallelize by the different number of topics.  A processor is allocated a value
  # of k, and does the cross-validation serially.  This is because it is assumed there
  # are more candidate values of k than there are cross-validation folds, hence it
  # will be more efficient to parallelise
  system.time({
    results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
      k <- candidate_k[j]
      print(k)
      results_1k <- matrix(0, nrow = folds, ncol = 2)
      colnames(results_1k) <- c("k", "perplexity")
      for(i in 1:folds){
        train_set <- full_data[splitfolds != i , ]
        valid_set <- full_data[splitfolds == i, ]
        
        fitted <- LDA(train_set, k = k, method = "Gibbs",
                      #control = list(alpha=eachalpha/k,burnin = burnin, iter = iter, keep = keep) )
                      control = list(verbose = 500,
                                     alpha=eachalpha))
        
        #fitted <- LDA(train_set, k = k, method = "Gibbs")
        
        results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
      }
      return(results_1k)
    }
  })
  stopCluster(cluster)
  
  results_df <- as.data.frame(results)
  results_df$myalpha<-as.character(eachalpha)
  MainresultDF<-rbind(MainresultDF,results_df)
  print ("DONE!!!")
  print(Sys.time())
}

MainresultDF$kalpha=paste0(as.character(MainresultDF$k),MainresultDF$myalpha) 

ggplot(MainresultDF) +
  geom_boxplot(aes(x=k, y=perplexity, group=kalpha,color=myalpha))+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.5)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.2)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.1)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.05)]),linetype = "dotted")+
  geom_hline(yintercept=min(MainresultDF$perplexity[which(MainresultDF$myalpha==0.01)]),linetype = "dotted")

alpha05 <- MainresultDF %>% 
  filter(myalpha == 0.05)


alpha05 <-alpha05[order(alpha05$k),]

cars.spl <- with(alpha05, smooth.spline(k, perplexity, df = 3))

plot(with(cars, predict(cars.spl, x = alpha05$k, deriv = 2)), type = "l")
abline(v=50)


alpha01 <- MainresultDF %>% 
  filter(myalpha == 0.01)

alpha01 <-alpha01[order(alpha01$k),]

cars.spl <- with(alpha01, smooth.spline(k, perplexity, df = 3))

plot(with(cars, predict(cars.spl, x = alpha01$k, deriv = 2)), type = "l")
abline(v=40)

runsdf<-data.frame(myk=c(40,50))

mymodels<-list()

cluster <- makeCluster(detectCores(logical = TRUE))
registerDoParallel(cluster)

clusterEvalQ(cluster, {
  library(topicmodels)
})

#clusterExport(cluster, c("full_data", "burnin", "iter", "keep", "splitfolds", "folds", "candidate_k"))
clusterExport(cluster, c("full_data","runsdf"))

system.time({
  mymodels <- foreach(j = 1:nrow(runsdf)) %dopar%{
    k_run <- runsdf[j,1]
    #alpha_run<-runsdf[j,2]
    fitted <- LDA(full_data, k = k_run, method = "Gibbs",
                  control = list(alpha=0.05, seed=267348))
    #control = list(seed=3341) )
  }
})
stopCluster(cluster)


data3 <- bernie2

LDAfit<-mymodels[[1]]
metadf<-data3

bernie2_short<-bernie2

missing_docs<-setdiff(dfm_ber@Dimnames$docs,LDAfit@documents)

bernie2_short<-bernie2_short[-which(bernie2_short$index %in% missing_docs),]
meta_theta_df<-cbind(bernie2_short,LDAfit@gamma)
dfm_short <- dfm_ber2
missing_docs2<-setdiff(dfm_short@Dimnames$docs,LDAfit@documents)
dfm_short <- dfm_short[-which(dfm_short@Dimnames$docs %in% missing_docs2), ]

dfm_forsize<-data.frame(dfm_short)
dfm_forsize<-dfm_forsize[,-1]


sizevect<-rowSums(dfm_forsize)
meta_theta_df<-data.frame(size=sizevect,meta_theta_df)

duplicate_df<-removed_df
colnames(duplicate_df)<-paste0(colnames(duplicate_df),".1")

dflist<-list()
for (i in (1:nrow(duplicate_df))) {
  the_match<-match(duplicate_df$text.1[i],meta_theta_df$text)
  newvect<-c(duplicate_df[i,],meta_theta_df[the_match,])
  dflist[[i]]<-newvect
}
maintable<-data.frame(do.call(bind_rows,dflist))

maintable<-data.frame(size=maintable$size,maintable[,-c((ncol(duplicate_df)+1):(ncol(duplicate_df)+ncol(metadf)+1))])
colnames(maintable)<-gsub("\\.1","",colnames(maintable))
meta_theta_df<-bind_rows(meta_theta_df,maintable)

meta_theta_df2 <- meta_theta_df %>% 
  filter(!size == "NA")

meta_theta_by_user<-aggregate(x = meta_theta_df2[,c(93:(ncol(meta_theta_df2)))], 
                              by = list(meta_theta_df2$screen_name), FUN = "mean")


meta_theta_by_user_retweets<-aggregate(x = meta_theta_df2[,c(15)], 
                                       by = list(meta_theta_df2$screen_name), FUN = "sum")


colnames(meta_theta_by_user_retweets)[colnames(meta_theta_by_user_retweets)=="Group.1"] <- "screen_name"
colnames(meta_theta_by_user_retweets)[colnames(meta_theta_by_user_retweets)=="x"] <- "retweets"

meta_theta_by_user_favourites<-aggregate(x = meta_theta_df2[,14], 
                                         by = list(meta_theta_df2$screen_name), FUN = "sum")

colnames(meta_theta_by_user_favourites)[colnames(meta_theta_by_user_favourites)=="Group.1"] <- "screen_name"
colnames(meta_theta_by_user_favourites)[colnames(meta_theta_by_user_favourites)=="x"] <- "favourites"

meta_theta_df2$forsum<-1
meta_theta_by_user_volume<-aggregate(x = meta_theta_df2[,"forsum"], 
                                     by = list(meta_theta_df2$screen_name), FUN = "sum")

meta_theta_by_user2 <- cbind(meta_theta_by_user, meta_theta_by_user_volume$x)

meta_theta_by_user2 <- cbind(meta_theta_by_user2, meta_theta_by_user_retweets$retweets)

meta_theta_by_user2 <- cbind(meta_theta_by_user2, meta_theta_by_user_favourites$favourites)


colnames(meta_theta_by_user2)[colnames(meta_theta_by_user2) == "meta_theta_by_user_volume$x"] <- "n"
colnames(meta_theta_by_user2)[colnames(meta_theta_by_user2) == "meta_theta_by_user_retweets$retweets"] <- "retweets"
colnames(meta_theta_by_user2)[colnames(meta_theta_by_user2) == "meta_theta_by_user_favourites$favourites"] <- "fav"

meta_theta_by_user3 <- meta_theta_by_user2[meta_theta_by_user2$n >= 20, ]

screen_name <- meta_theta_by_user3$Group.1

library(tweetbotornot2)

bots <- predict_bot(screen_name)

missing_name <- setdiff(screen_name, bots$screen_name)

bots2 <- predict_bot(missing_name) 

missing_name2 <- setdiff(missing_name, bots2$screen_name)

bots3 <- predict_bot(missing_name2)
  
bots4 <- rbind(bots, bots2, bots3)

colnames(meta_theta_by_user3)[colnames(meta_theta_by_user3)=="Group.1"] <- "screen_name"

meta_theta_by_user_bots <- meta_theta_by_user3 %>% 
  full_join(bots4)

meta_theta_by_user_bots$user_id <- NULL

meta_theta_by_user_bots2 <- meta_theta_by_user_bots %>% 
  mutate(bot = 
           ifelse(meta_theta_by_user_bots$prob_bot >= 0.70, 1,
                  ifelse(meta_theta_by_user_bots$prob_bot < 0.70, 2,
                         ifelse(meta_theta_by_user_bots$prob_bot == "NA", 3,
                                "NA"))))


themesbyuser_bots <- meta_theta_by_user_bots %>% 
  select(-n) %>% 
  select(-retweets) %>% 
  select(-fav) %>% 
  select(-prob_bot)

                       ###############################################################################
                       ####################                                     ######################
                       ####################    Thematic Communities Method      ######################
                       ####################  by Walter, Ophir & Jamieson(2020)  ######################
                       ####################                                     ######################
                       ###############################################################################

rownames(themesbyuser_bots)<-themesbyuser_bots$screen_name

themesbyuser_bots <- themesbyuser_bots[,-1]

themesbyuser_bots2 <- t(themesbyuser_bots)

library(lsa)
mycosine <- cosine(as.matrix(themesbyuser_bots2))


library(igraph)
sem_net_weighted<-graph.adjacency(mycosine,mode="undirected",weighted=T,diag=F,add.colnames="label")

V(sem_net_weighted)$name <- V(sem_net_weighted)$label
#V(sem_net_weighted)$followers <- themesbyuser_bots2$followers
V(sem_net_weighted)$size <- meta_theta_by_user_bots2$n
V(sem_net_weighted)$retw <- meta_theta_by_user_bots2$retweets
V(sem_net_weighted)$favourites <- meta_theta_by_user_bots2$fav
V(sem_net_weighted)$bots <- meta_theta_by_user_bots2$bot

set.seed(1983)

library(tidyverse)
library(skynet)

g<-disparity_filter(g=sem_net_weighted,alpha=0.33)

is.connected(g)
vcount(g)
ecount(g)

ecount(sem_net_weighted)

table(clusters(g)$membership)

set.seed(433547)
mylouvain<-(cluster_louvain(g))
mywalktrap<-(cluster_walktrap(g)) 
myinfomap<-(cluster_infomap(g)) 
myfastgreed<-(cluster_fast_greedy(g))
mylabelprop<-(cluster_label_prop(g))

V(g)$louvain<-mylouvain$membership 
V(g)$walktrap<-mywalktrap$membership 
V(g)$infomap<-myinfomap$membership  
V(g)$fastgreed<-myfastgreed$membership 
V(g)$labelprop<-mylabelprop$membership

nodelist<-list()
for (node in 1:length(V(g))) {
  #for (node in 1:100) {
  print(node)
  outside<-strength(g, vids = V(g)[node])
  tempg<-induced_subgraph(g,V(g)$louvain==V(g)$louvain[node])
  inside<-strength(tempg, vids = V(tempg)$label==V(g)[node]$label)
  nodelist[[node]]<-data.frame(
    node=node,label=V(g)[node]$label,inside,comm=V(g)$louvain[node],between=outside,within=inside,commstr=inside/outside)
}

user_comm_df<-do.call(rbind,nodelist)

##grab for each comm the top 20 users
top_user_com_df<-data.frame(matrix(NA, nrow = 20, ncol = length(unique(user_comm_df$comm))))

for (i in 1:max(user_comm_df$comm)) {
  print (i)
  temp_df<-user_comm_df[user_comm_df$comm==i,]
  temp_df<-temp_df[order(temp_df$commstr,decreasing = TRUE),]
  towrite<-temp_df$label[1:20]
  top_user_com_df[,i]<-towrite
}

comm_tweets_list<-list()
for (i in 1:max(user_comm_df$comm)) {
  print(i)
  temp_meta_theta_df<-meta_theta_df2[meta_theta_df2$screen_name %in% top_user_com_df[,i],]
  temp_meta_theta_df<- temp_meta_theta_df[sample(nrow(temp_meta_theta_df), 100), ]
  comm_tweets_list[[i]]<-c(temp_meta_theta_df)
  openxlsx::write.xlsx(temp_meta_theta_df,paste0(as.character(i),"_COMM_200_tweets.xlsx"))
}

library(tidyverse)

######################################## Network Analysis ########################################################
#Get the screen names and the mentions
mentions <- bernie %>% 
  select(screen_name, mentions_screen_name) 

#Next, since mentions_scree_names gives a list of names,
#I unnest all the mentioned user names to get two columns
#with only one name per row and column
mentions <- mentions %>%
  unnest(mentions_screen_name) %>% 
  filter(!is.na(mentions_screen_name))

#Now I get the retweets
bernie_rt <- bernie %>% 
  filter(is_retweet == T)

#And I get the names of the users and the names of the people
#that got retweeted
bernie_rt_net <- bernie_rt %>% 
  select(screen_name, retweet_screen_name)

#I delete NAs
poster_retweet <- na.omit(bernie_rt_net)

#change names of columns so both dfs have the same names
colnames(poster_retweet)[colnames(poster_retweet)== "retweet_screen_name"] <- "receiver"
colnames(mentions)[colnames(mentions)=="mentions_screen_name"] <- "receiver"

#create a df for both mentions and retweets
rt_ment <- tribble()

#join the dfs
rt_ment <- rt_ment %>% 
  bind_rows(poster_retweet, mentions)

#turn them into a matrix
rt_ment <- as.matrix(rt_ment)

library(igraph)

#and into an igraph object
g <- graph.edgelist(rt_ment)

#Now I calculate the eigenvector centrality
#And I turn the results into a df to later filter the users by 
#their centrality
eigen_cent <- eigen_centrality(g)
eigen_cent_users <- as.data.frame(eigen_cent[[1]])
eigen_cent_users$screen_name <- rownames(eigen_cent_users)
colnames(eigen_cent_users)[1] <- "eigen_cent"


#Since the network is very big I discard the 99.9% of users
#and keep only the top ones
top_5_eigen <- quantile(eigen_cent_users$eigen_cent, prob = .99)
eigen_cent_users$top_5_eigen_users <- ifelse(eigen_cent_users$eigen_cent >= top_5_eigen, eigen_cent_users$screen_name,NA)

#select those two columns
top_users <- eigen_cent_users %>% 
  select(eigen_cent, top_5_eigen_users)

#change the name of the column so they have the same one
colnames(top_users)[colnames(top_users)== "top_5_eigen_users"] <- "screen_name"

#and merge the dfs by sreen_name
poster_retweet <- left_join(as.data.frame(rt_ment), top_users, by = "screen_name")

#delete NAs
top_users2 <- na.omit(poster_retweet)

#and turn it into a matrix
top_users3 <- as.matrix(top_users2)

#and back into an igraph object again
g <- graph_from_data_frame(top_users3)

#I make it a weighted graph
wg <- g

E(wg)$weight <- runif(ecount(wg))

#I calculate the in-degree
degree_in <- sort(degree(wg, mode = "in"))

V(wg)$degree_in <- degree(wg, mode = "in")

#and the out-degree to maybe see who's retweeting and talking a lot
degree_out <- sort(degree(wg, mode = "out"))

V(wg)$degree_out <- degree(wg, mode = "out")

#to have a look at it
degree_in_df <- as.data.frame(degree_in)

degree_out_df <- as.data.frame(degree_out)

#calculate the strength
V(wg)$strength <- strength(wg, mode = "in")

#to have a look at it
strength <- sort(strength(wg))

strength_df <- as.data.frame(strength)

#I turn it into an undirected graph
und_net <-as.undirected(wg, mode= "collapse",
                        edge.attr.comb=list(weight="sum", "ignore"))


#the network has multiple edges and loops, I simplify it
und_net2 <- simplify(und_net, remove.multiple = T, remove.loops = T,
                     edge.attr.comb=c(weight="sum", type="ignore"))

#and run clustering algorithms
mylouvain <- cluster_louvain(und_net2)

V(und_net2)$louvain <- mylouvain$membership

mylabel <- cluster_label_prop(und_net2)

V(und_net2)$mylabel <- mylabel$membership

myspinglass <- cluster_spinglass(und_net2)

V(und_net2)$spinglass <- myspinglass$membership

myfastgreedy <- cluster_fast_greedy(und_net2)

V(und_net2)$myfastgreedy <- myfastgreedy$membership

myinfo <- cluster_infomap(und_net2)

V(und_net2)$info <- myinfo$membership

#I calculate the betweenness
V(und_net2)$btw <- betweenness(und_net2, v = V(und_net2), directed = FALSE)

#turn it into a df to see
between <- betweenness(und_net2, v = V(und_net2), directed = FALSE)

between_df <- as.data.frame(between)

between_df$screen_name <- rownames(between_df)

screen_name <- between_df$screen_name

#See if the users of the network are bots using tweetbotornot2
library(tweetbotornot2)
bots <- predict_bot(screen_name)

missing_name <- setdiff(screen_name, bots$screen_name)

bots2 <- predict_bot(missing_name)

bots3 <- rbind(bots, bots2)

between_with_bots <- between_df %>% 
  full_join(bots3)

between_with_bots2 <- between_with_bots %>% 
  mutate(bot = 
           ifelse(between_with_bots$prob_bot >= 0.70, 1,
                  ifelse(between_with_bots$prob_bot < 0.70, 2,
                         ifelse(between_with_bots$prob_bot == "NA", 3,
                                "NA"))))

V(und_net2)$bots <- between_with_bots2$bot

##Network Statistics
avg <- average.path.length(und_net2)

longest <- closeness(und_net2)

edge_density(und_net2)

transitivity(und_net2)

diameter(und_net2)

## K-Core Analysis
cores <- coreness(und_net2)

cores_df <- as.data.frame(cores)

V(und_net2)$core <- coreness(und_net2)

cores21 <- induced.subgraph(und_net2, V(und_net2)$core >= 21)

transitivity(cores21)

edge_density(cores21)

average.path.length(cores21)

eigen <- eigen_centrality(cores21)

eigen_df <- eigen[[1]]

eigen_df <- as.data.frame(eigen_df)

eigen_tot <- eigen_centrality(und_net2)

eigen_tot_df <- eigen_tot[[1]]

eigen_tot_df <- as.data.frame(eigen_tot_df)

coreslow <- induced.subgraph(und_net2, V(und_net2)$core <= 10)

transitivity(coreslow)

edge_density(coreslow)

average.path.length(coreslow)

eigen <- eigen_centrality(coreslow)

eigen_df <- eigen[[1]]

eigen_df <- as.data.frame(eigen_df)

write.graph(coreslow, "coreslow.RT.Mentions.Bots.graphml", format = "graphml")

write.graph(cores21, "core21.graphml", format = "graphml
