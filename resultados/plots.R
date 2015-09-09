setwd("~/Documents/cpp/tracking/resultados/")
library(ggplot2)
library(xtable)

dat<-read.csv("results-100-3-10.txt")
#dat<-dat[dat$reinit<400,]
dat$fps<-as.numeric(dat$fps)
dat$reinit<-as.numeric(dat$reinit)
dat$rate<-rep(0,dim(dat)[1])
dat$rate[dat$reinit!=0]<-100*dat$reinit[dat$reinit!=0]/dat$frames[dat$reinit!=0]
mm.dat<-aggregate(.~method+sequence,mean,data=dat[!dat$fps==Inf,])
print(xtable(mm.dat[,c(2,1,6,7,8)],digits=2),include.rownames=FALSE)

mm1<-mm.dat[mm.dat$method=="pmmh",]
mm2<-mm.dat[mm.dat$method=="tracker",]
mm3.dat<-merge(mm1,mm2,by="sequence")
names(mm3.dat)<-c("sequence","method","accuracy","recall","fps","reinit","frames","rate","method","accuracy","recall","fps","reinit","frames","rate")
print(xtable(mm3.dat[,c(1,5,8,12,15)],digits=2),include.rownames=FALSE)

mm2.dat<-aggregate(.~method,mean,data=dat[!dat$fps==Inf,])
print(xtable(mm2.dat[,c(1,3,4)],digits=2),include.rownames=FALSE)


hp<-ggplot(dat,aes(x=fps))
#hp<-hp+geom_histogram(alpha=.8,data=subset(dat,method=="pmmh"),binwidth=10,fill="light blue")
#hp<-hp+geom_histogram(alpha=.2,data=subset(dat,method=="tracker"),binwidth=10,fill="black")
hp<-hp+geom_histogram(binwidth=10,alpha=.8)+facet_grid(.~method)
hp<-hp+theme_bw()+scale_fill_manual(values=c("#999999", "#FFFFFF"))
ggsave(hp, file="fps.pdf")
dev.off()
hp<-ggplot(dat,aes(x=rate))
#hp<-hp+geom_histogram(alpha=.8,data=subset(dat,method=="pmmh"),binwidth=10,fill="light blue")
#hp<-hp+geom_histogram(alpha=.2,data=subset(dat,method=="tracker"),binwidth=10,fill="black")
hp<-hp+geom_histogram(binwidth=10,alpha=.8)+facet_grid(.~method)
hp<-hp+theme_bw()+scale_fill_manual(values=c("#999999", "#FFFFFF"))
ggsave(hp, file="reinit.pdf")
dev.off()
hp<-ggplot(dat,aes(x=accuracy))
#hp<-hp+geom_histogram(alpha=.8,data=subset(dat,method=="pmmh"),binwidth=0.1,fill="light blue")
#hp<-hp+geom_histogram(alpha=.2,data=subset(dat,method=="tracker"),binwidth=0.1,,fill="black")
hp<-hp+geom_histogram(binwidth=0.1,alpha=.8)+facet_grid(.~method)
hp<-hp+theme_bw()+scale_fill_manual(values=c("#999999", "#FFFFFF"))
ggsave(hp, file="accuracy.pdf")
dev.off()
hp<-ggplot(dat,aes(x=recall))
#hp<-hp+geom_histogram(alpha=.8,data=subset(dat,method=="pmmh"),binwidth=0.1,fill="light blue")
#hp<-hp+geom_histogram(alpha=.2,data=subset(dat,method=="tracker"),binwidth=0.1,,fill="black")
hp<-hp+geom_histogram(binwidth=0.1,alpha=.8)+facet_grid(.~method)
hp<-hp+theme_bw()+scale_fill_manual(values=c("#999999", "#FFFFFF"))
ggsave(hp, file="recall.pdf")
dev.off()

s<-unique(dat$sequence)
for(i in 1:length(s)){
    m<-dat[dat$sequence==s[i],]
    hp<-ggplot(m,aes(x=method,y=reinit,colour="white"))+geom_boxplot(colour="black")
    hp<-hp + theme_bw()+ggtitle(s[i])
    ggsave(hp, file=paste0(s[i],".pdf"))
    dev.off()
}