rm(list=ls())

library(ggplot2)

K = 3

cluster_data <- read.csv(paste('./gmm_em.csv', sep=""))

mu1 <- cluster_data[cluster_data$cluster == 0, c('mu1', 'mu2')][1,]
mu2 <- cluster_data[cluster_data$cluster == 1, c('mu1', 'mu2')][1,]
mu3 <- cluster_data[cluster_data$cluster == 2, c('mu1', 'mu2')][1,]

mu_data <- rbind(mu1, mu2, mu3)


p <- ggplot(data=cluster_data, aes(x1, x2))
#p <- ggplot(data=cluster_data, aes(x1, x2, color=factor(cluster)))
#p <- p + scale_color_brewer(type='qual')
p <- p + geom_point()
# p <- p + geom_point(data=mu_data, aes(mu1, mu2, color='red'))
# p <- p + scale_size_manual(values=rep(10,3))
p <- p + stat_spoke(data=cluster_data, aes(angle=c(r1,r2)*2*pi, radius=0.01))
print(p)