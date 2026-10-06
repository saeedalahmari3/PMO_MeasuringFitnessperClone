# conda activate r_env
# setwd("/raid/crdlab/ix1/Projects/M005_MeasuringFitnessPerClone_2019/code/R")
# setwd("/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging/R")
setwd("/Users/saeedalahmari/Documents/BioInformaticsPaper/PMO_MeasuringFitnessperClone/3D_Imaging/R")
source("CorrectCellposeSegmentation.R")
source("assignCompartment2Nucleus.R")
source("compareCells.R")
# source("clusterMito.R")
source("generateImageMask.R")
source("Utils.R")
source("visualizeSingleCells.R")
eps=read.table('../dbscan/eps.txt')
library(abind)
library(matlab)
library(rgl)
library(ijtiff)
library(geometry)
library(habtools)
library(flexclust)
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")


## Constants, Settings, Input and output folders:
#ROOT="/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data/NCI-N87"
# ROOT="/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data/NCI-N87"
ROOT="//Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/NCI-N87-combined"
setwd(ROOT)
EPS=10; #6
MINPTS=3; #4
MAXDIST2FUCCISIGNAL=3
xydim = 255
pixelsize_xy = 0.232 # um 
z_interval = 0.29 #  um 
xyz=c("x","y","z")
# r3dDefaults$windowRect=c(0,50, 1600, 800) 
r3dDefaults$windowRect=c(0,50, 500,500)
INDIR="A04_CellposeOutput"
OUTCORRECTED="A05_PostProcessCellposeOutput"
dirCreate(OUTCORRECTED, permission = "a+w")
# ILASTIKINPUT="G06_IlastikInput"
# dirCreate(ILASTIKINPUT, permission = "a+w")
OUTLINKED="A06_multiSignals_Linked"
OUTSTATS="K07_LinkedSignals_Stats"
OUTPSEUDOTIME="K08_Pseudotime"
FUCCIDIR = "I08_3DCellProfiler_FUCCI"
DATA4PAPERDIR="~/Repositories/CellCycle4DataIntegration/data4paper/A08_Pseudotime"
dirCreate(OUTPSEUDOTIME, permission = "a+w")
dirCreate(OUTLINKED, permission = "a+w")
dirCreate(OUTSTATS, permission = "a+w")
xmlfiles=list.files('A01_rawData/',pattern=".xml",full.names=T)


## Local helper functions
correctSegmentations<-function(FoF, signals, eps){
  sapply(names(signals), function(x) CorrectCellposeSegmentation(FoF,signal=x,INDIR,OUTCORRECTED,doplot=F,eps=eps[FoF,x]))
}
readOrganelleCoordinates<-function(signals_per_id, signals, IN){
  coord=NULL;
  for(cell in signals_per_id$x){
    for(s in signals){
      x=paste0(s,"_cell_",cell,"_coordinates.csv")
      a=read.csv(paste0(IN,filesep,x))
      ## pixel to um conversion: @TODO
      a$z=a$z*z_interval
      a$x=a$x*pixelsize_xy
      a$y=a$y*pixelsize_xy
      # id=strsplit(fileparts(x)$name,"_")[[1]]
      a$organelle = a[,ncol(a)] 
      a$intensity = a$signal
      a$signal=s
      # id=id[length(id)-1]
      a$id=cell
      ## add missing columns
      if(!is.null(coord)){
        for(mc in setdiff(colnames(a), colnames(coord))){
          coord[,mc]=NA
        }
        tmp=matrix(NA,nrow(a), ncol(coord))
        colnames(tmp) = colnames(coord)
        coord=rbind(tmp, coord)
        coord[1:nrow(a),colnames(a)] = a
      }else{
        coord=a
      }
      # a[,c("y", "x", "z", "organelle", "id","signal")]
    }
  }
  coord$id=as.numeric(coord$id)
  return(coord)
}


#REGEX="240918_fluorescent.nucleus"
#REGEX="2410"
#REGEX="231005_fluorescent.nucleus"



#REGEX="231005"
REGEX="2409"
#REGEX="231005_fluorescent.nucleus"
if (grepl("^2409", REGEX) || grepl("^2410",REGEX)){
  MINGREEN = 0.003
  MINRED = 0.0095
  MAXRED = 0.011
  print(REGEX)
  print(sprintf("MINGREEN is %f MINRED is %f", MINGREEN,MINRED))
}else if(grepl("^231005", REGEX)){
  MINGREEN = 0.005
  MINRED = 0.009
  MAXRED = 0.01
  print(REGEX)
  print(sprintf("MINGREEN is %f MINRED is %f", MINGREEN,MINRED))
}

##########################
### read fucci results ###
# FUCCIDIR=paste0("/Volumes/Expansion/Collaboration/Moffitt_Noemi/BioinformaticsPaper/data")
f=list.files(FUCCIDIR, recursive=T, pattern="object_features_new\\.csv$", full.names=T) #object_240918.csv
fucciFile=grep(REGEX, f, value=T)
if(length(fucciFile)==0){
  stop(paste("No FUCCI object_features.csv file matched REGEX", REGEX,
             "under", FUCCIDIR, "\nAvailable files:\n",
             paste(f, collapse="\n")))
}
if(length(fucciFile)>1){
  warning(paste("Multiple FUCCI object_features.csv files matched REGEX", REGEX,
                "; using", fucciFile[1]))
}
fucci=read.csv(fucciFile[1], check.names=F, stringsAsFactors=F)
ii=which(colnames(fucci) %in% toupper(xyz))
colnames(fucci)[ii]=tolower(colnames(fucci)[ii])
## Plot fucci classes
# par(mfrow=c(2,2))



## classify cell cycle phase
fucci$cellCycle = 2
fucci$cellCycle[fucci$Intensity_MeanIntensity_green>MINGREEN & fucci$Intensity_MeanIntensity_red<MINRED] = 1
fucci$cellCycle[fucci$Intensity_MeanIntensity_green<MINGREEN & fucci$Intensity_MeanIntensity_red>MINRED] = 3
fucci$cellCycle[fucci$Intensity_MeanIntensity_green>MINGREEN & fucci$Intensity_MeanIntensity_red>MINRED] = 4

fucci <- fucci[!(fucci$cellCycle == 4 & fucci$Intensity_MeanIntensity_red > MINRED & fucci$Intensity_MeanIntensity_red <= MAXRED), , drop=FALSE]
#fucci$cellCycle[fucci$cellCycle=="G2M"] = 4
#fucci$cellCycle[fucci$cellCycle=="S"] = 3
#fucci$cellCycle[fucci$cellCycle=="G1S"] = 2
#fucci$cellCycle[fucci$cellCycle=="G1"] = 1   
#fucci$cellCycle=as.numeric(fucci$cellCycle)   

fuccicol=1:4
names(fuccicol) = c("G1", "G1/S","S","G2/M")
plot(fucci$Intensity_MeanIntensity_green, fucci$Intensity_MeanIntensity_red, col=fucci$cellCycle, pch=20, log="xy", xlab="Intensity_IntegratedIntensity_green", ylab="Intensity_IntegratedIntensity_red", cex.lab=2, cex.axis=2, main=REGEX)
#with(subset(fucci, FoF == 2), plot(Intensity_MeanIntensity_green, Intensity_MeanIntensity_red, col=cellCycle, pch=20, log="xy", xlab="Intensity_IntegratedIntensity_green", ylab="Intensity_IntegratedIntensity_red", cex.lab=2, cex.axis=2, main=paste(REGEX, "FoF2")))
legend("bottomright", names(fuccicol), fill=fuccicol)
plyr::count(fucci$cellCycle)
## Save output for gating
coi=sapply(c("red","green"), function(x) grep(x, colnames(fucci), value=T))
coi=c(coi[,1], coi[,2])
coi=grep("Name_", coi, invert = T, value = T)
coi=grep("Intensity", coi, value = T)
#fucci$cellCycle = NA
write.table(fucci[,coi],file = paste0("~/Downloads/",REGEX,"_fucci.txt"),row.names = F,quote = F)

#Save csv file with cellCyles

fucciOut <- fucci

cellCycleMap <- c(
  "1" = "G1",
  "2" = "G1S",
  "3" = "S",
  "4" = "G2M"
)

fucciOut$cellCycle <- unname(cellCycleMap[as.character(fucciOut$cellCycle)])

# Ensure cellCycle is the last column
fucciOut <- fucciOut[
  , c(setdiff(colnames(fucciOut), "cellCycle"), "cellCycle"),
  drop = FALSE
]

outputFile <- sub(
  "\\.csv$",
  "_withCellCycle.csv",
  fucciFile[1],
  ignore.case = TRUE
)

write.csv(
  fucciOut,
  file = outputFile,
  row.names = FALSE,
  quote = FALSE
)

message("Saved: ", outputFile)

###############################################
######Allen model performance evaluation#######
###############################################

## Input and output:
# FoFs=list.files(INDIR, pattern=REGEX)
FoFs=list.files(INDIR, pattern=REGEX)
signals=list(nucleus.p="nucleus.p_Cells_Centers.csv");#,mito.p="mito.p_Cells_Centers.csv", cytoplasm.p="cytoplasm.p_Cells_Centers.csv")
# signals=list(nucleus.p="nucleus.p_Cells_Centers.csv"); #nucleus.t="nucleus.t_Cells_Centers.csv",
# signals=list(nucleus.t="nucleus.t_Cells_Centers.csv"); 
stats=list()
mfrow3d(nr = 1, nc = 4, sharedMouse = TRUE)  
rawimges <- images <- list()
ncells=list()
for(FoF in FoFs){
  print(FoF)
  setwd(ROOT)
  unlink(paste0(OUTCORRECTED,filesep,FoF),recursive=T)
  
  ###############################################
  ###### Correcting Cellpose Segmentation #######
  ###############################################
  CorrectCellposeSegmentation(FoF,signal=names(signals)[1],INDIR,OUTCORRECTED,doplot=0,eps=EPS,minPts=MINPTS,IMPORTALLORGANELLES=F)
  #OUTLINKED_=paste0(getwd(),filesep,OUTCORRECTED,filesep,FoF,filesep,"All_Cells_coordinates")
  ncells[[FoF]]= length(list.files(paste0(OUTCORRECTED,filesep,FoF,filesep,"All_Cells_coordinates"),pattern = "nucleus"))
  # # ## For live-cell tracking:
  # # # # CorrectCellposeSegmentation(FoF,signal=names(signals),INDIR,OUTCORRECTED,doplot=F,eps=EPS,minPts=MINPTS,IMPORTALLORGANELLES=F)
  # # # # rgl.snapshot("~/Downloads/Brightfield_Timeseries.png")
  # # # images[[FoF]]=generateImageMask(FoF, INDIR=OUTCORRECTED, OUTDIR=OUTCORRECTED,root = ROOT, xydim = xydim)
  # images[[FoF]]=try(generateImageMask(FoF, INDIR=OUTCORRECTED, OUTDIR=OUTCORRECTED,root = ROOT, signal = "nucleus.p"))
  # # # img=bioimagetools::readTIF(paste0(INDIR,filesep,FoF,filesep,names(signals),".tif"))
  # # # img=img[fliplr(1:nrow(img)),,1:dim(images[[FoF]])[3]]
  # # # img=EBImage::rotate(img,-90)
  # # # rawimges[[FoF]]=resize4Ilastik(img, xydim = xydim)
  # 
  
  # ## Cluster Organelles
  # setwd(ROOT)
  # OUTSEG_=paste0(getwd(),filesep,OUTCORRECTED,filesep,FoF,filesep)
  # ##@TODO rm all mito and cyto files
  MITOTIF=paste0(ROOT,filesep,INDIR,filesep,FoF,"/mito.p.tif")
  # clusterMito(MITOTIF, OUTSEG_)
  CYTOTIF=paste0(ROOT,filesep,INDIR,filesep,FoF,"/cytoplasm.p.tif")
  # clusterMito(CYTOTIF, OUTSEG_)
  
  # ## For linking organelles or to link each predicted nucleus to its closest target nucleus
  setwd(ROOT)
  OUTLINKED_=paste0(getwd(),filesep,OUTLINKED,filesep,FoF,filesep)
  unlink(OUTLINKED_,recursive=T)
  setwd(paste0(OUTCORRECTED,filesep,FoF,filesep,"Cells_center_coordinates"))
  # # # assignCompartment2Nucleus(signals$nucleus.p, signals$nucleus.t, OUTLINKED_)
  # assignCompartment2Nucleus(signals$mito.p, signals$nucleus.p, OUTLINKED_, save_cell_gif=F)
  # assignCompartment2Nucleus(signals$cytoplasm.p, signals$nucleus.p, OUTLINKED_, save_cell_gif=F)
  assignCompartment2Nucleus(MITOTIF, CYTOTIF,OUTLINKED_, signals$nucleus.p, save_cell_gif=F)
  setwd(ROOT)
}
  
  ### RUN THE LOOP UP TO THIS POINT, THEN RUN CELLPROFILER. 
  ## Save FUCCI intensities
  ## Use Nucleus coordinates (from cellpose postprocess) to access intensities in ch00 and ch02 of FUCCI image:
for(FoF in FoFs){
  print(FoF)
  setwd(ROOT)
  OUTLINKED_=paste0(getwd(),filesep,OUTLINKED,filesep,FoF,filesep)
  fucci_coord=list()
  for(ch in c("ch00","ch02")){
    TIF=list.files(paste0("A01_rawData",filesep,FoF), pattern=ch, full.names = T)
    ch_=sapply(TIF, function(x) bioimagetools::readTIF(x), simplify = F)
    fucci_coord[[ch]]=do.call(abind, ch_)
  }
  f=list.files(OUTLINKED_, pattern="nucleus.p_cell_",full.names = T)
  ## Save cellprofiler output with matching nucleus ID:
  fucci_=fucci[fucci$FileName_bright==FoF,]
  fucci_$dist2nuc <- fucci_$ID <- NA
  for(x in f){
    coord=read.csv(file=x,check.names = F,stringsAsFactors = F)
    coord_ = coord
    coord_[,xyz] = coord_[,xyz]+1
    coord$fucci_ch00=apply(coord_,1, function(p) fucci_coord$ch00[p["y"],p["x"],p["z"]])
    coord$fucci_ch02=apply(coord_,1, function(p) fucci_coord$ch02[as.numeric(p["y"]),as.numeric(p["x"]),as.numeric(p["z"])] )
     ## Assign FUCCI ID:
    d=dist2(t(as.matrix(apply(coord_[,xyz],2,mean))), fucci_[,xyz])
    fucci_$ID[which.min(d)] = as.numeric(strsplit(fileparts(x)$name,"_")[[1]][3])
    fucci_$dist2nuc[which.min(d)] = min(d);
                           
    coord$cellCycle = fucci_$cellCycle[which.min(d)]
    coord$dist2fucci = min(d);
    ## Save output -- overwrite
    write.csv(coord, file=x,quote = F,row.names = F)
  }
  fucci_=fucci_[!is.na(fucci_$ID),]
  rownames(fucci_)=fucci_$ID
  write.table(fucci_, file=paste0(OUTLINKED_,filesep,FoF,"_fucci.txt"), quote = F,row.names = F, sep="\t")
  
  # ## Visualize cells
  # cells=unique(sapply(strsplit(list.files(OUTLINKED_,pattern = "nucleus.p"),"_"),"[[",3))
  # tmp=sapply(cells, function(i) visualizeSingleCells(i, signals$mito.p, signals$nucleus.p, OUTLINKED_))
  
  # ## Compare each predicted to its linked target nucleus
  # stats[[FoF]]=compareCells(signals$nucleus.t, signals$nucleus.p, OUTLINKED_)
}


## keep only cells with all three signals:
OUTLINKED="A06_multiSignals_Linked"
FoFs=list.files(OUTLINKED);
signals=list(nucleus.p="nucleus.p_Cells_Centers.csv",mito.p="mito.p_Cells_Centers.csv",cytoplasm.p="cytoplasm.p_Cells_Centers.csv")
signals_per_id=list()
## Input and output:
for(FoF in FoFs){
  OUTLINKED_=paste0(getwd(),filesep,OUTLINKED,filesep,FoF,filesep)

  ## Exclude cells where one or more signals are missing
  f=list.files(OUTLINKED_,full.names = T, pattern = ".csv")
  f_= sapply(f, function(x) fileparts(x)$name)
  signals_per_id_=plyr::count(sapply(strsplit(f_,"_"), function(x) x[length(x)-1]))
  toRM=signals_per_id_$x[signals_per_id_$freq<length(signals)]

  ## Exclude cells with ambiguous fucci signal
  rownames(signals_per_id_)=signals_per_id_$x
  fucci_=read.table(list.files(OUTLINKED_,full.names = T, pattern = "fucci.txt"),header = T)
  noFucciIDs=fucci_$ID[fucci_$dist2nuc>MAXDIST2FUCCISIGNAL]
  toRM=union(toRM, signals_per_id_[noFucciIDs,"x"])
    
  ## Now remove files
  for(x in toRM){
    y=list.files(OUTLINKED_,full.names = T,pattern = paste0("_",x,"_"))
    file.remove(y)
  }
  signals_per_id[[FoF]]=signals_per_id_[!signals_per_id_$x %in% toRM,]
}
barplot(sapply(signals_per_id,nrow), names=names(signals_per_id))

###########################
## Calculate image stats ##
###########################
for(FoF in FoFs){
  print(FoF)
  OUTLINKED_=paste0(getwd(),filesep,OUTLINKED,filesep,FoF,filesep)
  if(length(signals)==1){
    OUTLINKED_=paste0(getwd(),filesep,OUTCORRECTED,filesep,FoF,filesep,"All_Cells_coordinates")
  }
  
  ## Read in linked organelles
  coord_=readOrganelleCoordinates(signals_per_id[[FoF]], names(signals), OUTLINKED_)
  save(file=paste0("~/Downloads/stats_",FoF,".RObj"),"coord_")
  coord=coord_; ## Backup
  
  ## Gather stats
  cells=unique(coord_$id)
  thesignals=unique(coord_$signal)
  imgStats=as.data.frame(matrix(NA,length(cells),3+4*length(thesignals)))
  rownames(imgStats)=as.character(cells)
  colnames(imgStats)=c(sapply(c("vol_","area_","pixels_","count_"),paste0,thesignals),"x","y","z")
  imgStats$FUCCI_cellCycle=NA
  for(id in cells){
    print(paste("Processing",which(cells==id),"out of",length(cells),"cells ..."))
    # print(paste("cell",id))
    a=coord_[coord_$id==id,]
    imgStats[as.character(id),c("x","y","z")]=apply(a[a$signal=="nucleus.p",c("x","y","z")],2,median)
    ## shotcut to get convex hull stats
    emptyList=list(area=NA, vol=NA, maxAreaSlice=NA,fractaldim=NA,rugosity=NA,heightRange=NA, convexity=NA, packing=NA, sphericity=NA, sma=NA,csf=NA)
    getstats<-function(a_){
      hull <- try(convhulln(a_[,c("x","y","z")], options = "FA", return.non.triangulated.facets=T),silent = T)
      # hull=try(Plot_ConcaveHull(a_$x, a_$y, a_$z, lcolor =1, alpha=0.15,add = F))
      if(class(hull)=="try-error"){
        hull=emptyList
      }else{
        cmap <- try(convhulln(a_[,c("x","y","z")], options = "FA", return.non.triangulated.facets=F),silent = T)
        cmap=to.mesh3d(cmap)
        hull=hull[c("area","vol")]
        hull$heightRange=hr(cmap)
        # surface_area(cmap)
        hull$convexity=convexity(cmap)
        hull$packing=packing(cmap)
        hull$sphericity=sphericity(cmap)
        hull$sma=sma(cmap)
        hull$smv=smv(cmap)
        hull$csf=csf(mcap)
      }
      ## includes nucleus coordinates
      if(!isempty(grep("nucleus", a_$signal))){
        ## Central slice area
        slice <- try(convhulln(a_[,c("x","y")], options = "FA"),silent = T)
        hull$maxAreaSlice=slice$area
        # hull$fractaldim=fd(cmap, method = "cubes", plot = TRUE, diagnose = TRUE)$D
        hull$rugosity=rg(cmap)
      }
      hull$pixels=nrow(a_)
      return(hull)
    }
    ## stats for entire cell (all organelles together)
    stats_=getstats(a)
    stats_$count=1
    imgStats[as.character(id),paste0(names(stats_),"_cell")]=sapply(stats_,sum,na.rm=T)
    ## iterate through signals: mito, cyto, nucleus
    for(i in 1:length(thesignals)){
      signal=thesignals[i]
      if(isempty(grep("nucleus",signal))){
        stats_=emptyList
        for(organelle in unique(a$organelle)){
          a_=a[a$organelle==organelle & a$signal==signal,]
          hull=getstats(a_)
          for(sname in names(hull)){
            stats_[[sname]]=c(stats_[[sname]],hull[[sname]])
          }
        }
        stats_$count=length(stats_$area)-1
      }else{
        a_=a[a$signal==signal,]
        stats_=getstats(a_)
        stats_$count=1
      }
      imgStats[as.character(id),paste0(names(stats_),"_",signal)]=sapply(stats_,sum,na.rm=T)
      
      ## distance to other organelles
      a_=a[a$signal==thesignals[i],]
      a_= t(as.matrix(apply(a_[,c("x","y","z")], 2, median)))
      for(j in i:length(thesignals)){
        if(i==j){
          next
        }
        b_=a[a$signal==thesignals[j],]
        b_=grpstats(b_[,c("x","y","z")], b_$organelle,"median")$median
        d=dist2(a_,b_)
        imgStats[as.character(id), paste0(c("Min","Max","Median"),"Dist_",thesignals[i],"_",thesignals[j])]=quantile(d,c(0,1,0.5))
      }
    }
    
    ## intensity based features
    for(signal in grep("nucleus", thesignals, invert = T, value = T) ){
      a_=a[a$signal==signal,]
      imgStats[as.character(id),paste0(c("meanIntensity_","medianIntensity_","maxIntensity_","minIntensity_"),signal)]=c(mean(a_$intensity),median(a_$intensity),max(a_$intensity),min(a_$intensity));
    }
    
    ## Record fucci derived cell cycle class too:
    a_=a[a$signal==grep("nucleus", thesignals, value = T),]
    imgStats[as.character(id),"FUCCI_cellCycle"] =unique(a_$cellCycle)
  }
  
  ## Add more stats and save 
  if("mito.t" %in% names(signals) || "mito.p" %in% names(signals)){
    imgStats$pixel_per_mito_avg=imgStats$pixels_mito.p/imgStats$count_mito.p
    imgStats$pixel_per_volume_mito=imgStats$pixels_mito.p/imgStats$vol_mito.p
    imgStats$nuc_to_mito = imgStats$vol_nucleus.p/imgStats$vol_mito.p
  }
  for(organelle in thesignals){
    imgStats[,paste0("pixels_per_volume_",organelle)]=imgStats[,paste0("pixels_",organelle)]/imgStats[,paste0("vol_",organelle)]
  }
  if("cytoplasm.t" %in% names(signals)){
    imgStats$nuc_to_mito_plus_cyto = imgStats$vol_nucleus.p/(imgStats$vol_mito.p+imgStats$vol_cytoplasm.t)
    imgStats$nuc_to_cyto = imgStats$vol_nucleus.p/imgStats$vol_cytoplasm.t
    imgStats$cyto_to_mito = imgStats$vol_cytoplasm.t/imgStats$vol_mito.p
  }
  imgStats$nuc_vol_to_area=imgStats$vol_nucleus.p/imgStats$area_nucleus.p
  write.table(imgStats,file=paste0(OUTSTATS,filesep,FoF,"_stats.txt"),sep="\t",quote=F,row.names = T)
  
  
  ## Visualize stats
  imgStats=read.table(paste0(OUTSTATS,filesep,FoF,"_stats.txt"),header = T,sep="\t")
  ## Z-score
  imgStats_=apply(imgStats, 2, function(x) (x - mean(x,na.rm=T))/sd(x,na.rm=T))
  imgStats_=imgStats_[,!apply(is.na(imgStats_),2,any)]
  jj=grep("nuc", colnames(imgStats_), value = T, ignore.case = T)
  jj=grep("mito", jj, value = T, invert = T, ignore.case = T)
  jj=c("rugosity_nucleus.p"      ,         "smv_nucleus.p"   ,                 "convexity_nucleus.p"        , "sphericity_nucleus.p" )
  tmp=as.matrix(imgStats_[,jj])
  tmp[!is.finite(tmp)]=NA
  hm = gplots::heatmap.2(tmp,trace = "none", margins = c(13, 6), symm = F)
}



##################
## Combine all ###
imgStats <- list()
for(FoF in FoFs){
  imgStats_=read.table(paste0(OUTSTATS,filesep,FoF,"_stats.txt"),sep="\t",check.names = F,stringsAsFactors = F,header = T)
  imgStats_$FoF= FoF
  ## which frame (timepoint) is this? Used for mapping to Ilastik output:
  imgStats_$ID=rownames(imgStats_)
  imgStats[[FoF]]=imgStats_
}
imgStats=do.call(rbind, imgStats)
imgStats=imgStats[,which(apply(imgStats, 2, function(x) !all(x==0 | is.na(x))))]
write.table(imgStats, paste0(OUTPSEUDOTIME,filesep,"LabelFree_stats.txt"),sep="\t",quote = F, row.names = T)
file.copy(paste0(OUTPSEUDOTIME,filesep,"LabelFree_stats.txt"),DATA4PAPERDIR, overwrite = T)
