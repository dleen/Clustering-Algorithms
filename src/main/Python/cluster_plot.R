rm(list=ls())

library(ggplot2)

K = 2

clusters <- read.csv(paste('./lloyds_',K,'_clusters.csv', sep=""))

pdf(paste('K',K,'.pdf',sep=""))

p <- ggplot(data=clusters, aes(x1, x2, color=factor(cluster)))
#p <- p + scale_color_brewer(type='qual')
p <- p + geom_point()
print(p)
                     
dev.off()