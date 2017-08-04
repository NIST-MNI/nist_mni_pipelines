library(ggplot2)
library(jsonlite)
library(grid)
library(plyr)
theme_set(theme_bw(base_size = 14, base_family = "Arial"))

cv_slow<-fromJSON("test_cv/cv_stats.json") 
cv_fast<-fromJSON("test_cv_fast/cv_stats.json") 
cv_lin<-fromJSON("test_cv_lin/cv_stats.json") 
cv_v_slow<-fromJSON("test_cv_slow/cv_stats.json")
cv_v_slow_1<-fromJSON("test_cv_slow_1/cv_stats.json")

cv_kappa<-data.frame(
  cv_slow=cv_slow$gkappa,
  cv_fast=cv_fast$gkappa,
  cv_lin=cv_lin$gkappa,
  cv_v_slow=cv_v_slow$gkappa,
  cv_v_slow_1=cv_v_slow_1$gkappa
    )

cvv<-stack(cv_kappa)

names(cvv)=c('GenKappa','Method')

png('cv_kappa.png',width=800,height=400,type='cairo')

ggplot(data=cvv,aes(x=Method,y=GenKappa))+
 geom_boxplot(notch=T)+
 theme_bw()+
 theme(
   axis.text = element_text(face = 'bold', vjust = 0.2, size = 18),
   axis.title = element_text(face = 'bold', vjust = 0.2, size = 20),
   plot.margin = unit(c(0.2,2.8,0.2,0.2), "cm")
   )

   
slen=length(names(cv_slow$result))
lcv <- vector(mode = "list", length = slen)

for(l in seq(slen)) {
  i=names(cv_slow$result)[l]
  cv_grading<-data.frame(
    grad_slow=cv_slow$result[,i]$grad,
    grad_fast=cv_fast$result[,i]$grad,
    grad_lin=cv_lin$result[,i]$grad,
    grad_v_slow=cv_v_slow$result[,i]$grad,
    grad_v_slow_1=cv_v_slow_1$result[,i]$grad
  )
  lcv[[l]]<-stack(cv_grading)
  names(lcv[[l]])=c('Grading','Method')
  lcv[[l]]$group=rep(cv_slow$group,length(names(cv_grading)))
  lcv[[l]]$struct=rep(i,length(lcv[[l]]$group))
}

cvv<-rbind.fill(lcv)
cvv$struct<-as.factor(as.numeric(cvv$struct))
cvv$group<-as.factor(cvv$group)

png('cv_grading.png',width=800,height=800,type='cairo')

ggplot(data=cvv,aes(x=group,y=Grading,colour=Method))+
 geom_boxplot(notch=T)+
 theme_bw()+
 facet_grid(struct~Method)+
 geom_abline(intercept=0,slope=0,colour='red',lty=2)+
 theme(
   axis.text = element_text(face = 'bold', vjust = 0.2, size = 18),
   axis.title = element_text(face = 'bold', vjust = 0.2, size = 20),
   plot.margin = unit(c(0.2,2.8,0.2,0.2), "cm")
   )
