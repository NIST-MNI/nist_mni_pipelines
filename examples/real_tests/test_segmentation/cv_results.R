library(ggplot2)
library(jsonlite)
library(grid)
theme_set(theme_bw(base_size = 14, base_family = "Arial"))


cv1<-fromJSON("test_cv/cv_1_stats.json") 
cv2<-fromJSON("test_cv/cv_2_stats.json") 
cv3<-fromJSON("test_cv/cv_3_stats.json") 
cv4<-fromJSON("test_cv/cv_4_stats.json") 

cv<-data.frame(
  cv1=cv1$gkappa,
  cv2=cv2$gkappa,
  cv3=cv3$gkappa,
  cv4=cv4$gkappa
    )


cvv<-stack(cv)

names(cvv)=c('GenKappa','Method')

png('cv.png',width=800,height=400,type='cairo')

ggplot(data=cvv,aes(x=Method,y=GenKappa))+
 geom_boxplot(notch=T)+
 theme_bw()+
 theme(
   axis.text = element_text(face = 'bold', vjust = 0.2, size = 18),
   axis.title = element_text(face = 'bold', vjust = 0.2, size = 20),
   plot.margin = unit(c(0.2,2.8,0.2,0.2), "cm")
   )
