#-*- coding: utf-8 -*-
"""
@author:MD.Nazmuddoha Ansary
"""
from __future__ import print_function

#-------------------------
# imports
#-------------------------
from .yolo import YOLO
from .utils import localize_box,LOG_INFO,download
from .rotation import auto_correct_image_orientation
from .paddet import Detector
from .checks import processNID,processDob
from paddleocr import PaddleOCR
import os
import cv2
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
#-------------------------
# class
#------------------------

    
class OCR(object):
    def __init__(self,   
                 yolo_onnx="weights/yolo.onnx",
                 yolo_gid="1gbCGRwZ6H0TO-ddd4IBPFqCmnEaWH-z7"):
        
        if not os.path.exists(yolo_onnx):
            download(yolo_gid,yolo_onnx)
        self.loc=YOLO(yolo_onnx,
                      labels=['sign', 'bname', 'ename', 'fname', 
                              'mname', 'dob', 'nid', 'front', 'addr', 'back'])
        LOG_INFO("Loaded YOLO")
        
        self.base=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet',use_gpu=True)
        self.det=Detector()
        LOG_INFO("Loaded Paddle")

        
    def process_boxes(self,text_boxes,region_dict,includes):
        '''
            keeps relevant boxes with respect to region
            args:
                text_boxes  :  detected text boxes by the detector
                region_dict :  key,value pair dictionary of region_bbox and field info 
                               => {"field_name":[x_min,y_min,x_max,y_max]}
                includes    :  list of fields to be included 
        '''
        # extract region boxes
        region_boxes=[]
        region_fields=[]
        for k,v in region_dict.items():
            if k in includes:
                region_fields.append(k)
                region_boxes.append(v)
        # ref_boxes
        ref_boxes=[]
        for bno in range(len(text_boxes)):
            tmp_box = copy.deepcopy(text_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            ref_boxes.append([x1,y1,x2,y2])
        # sort boxed
        data=pd.DataFrame({"ref_box":ref_boxes,"ref_ids":[i for i in range(len(ref_boxes))]})
        # detect field
        data["field"]=data.ref_box.apply(lambda x:localize_box(x,region_boxes))
        data.dropna(inplace=True) 
        data["field"]=data["field"].apply(lambda x:region_fields[int(x)])
        box_dict={}

        for field in data.field.unique():
            _df=data.loc[data.field==field]
            boxes=_df.ref_box.tolist()
            idxs =_df.ref_ids.tolist()
            idxs=[x for _, x in sorted(zip(boxes,idxs), key=lambda pair: pair[0][0])]
            box_dict[field]=idxs

        return box_dict

    
    #-------------------------------------------------------------------------------------------------------------------------
    # exectutives
    #-------------------------------------------------------------------------------------------------------------------------
    def execute_rotation_fix(self,image,mask):
        image,mask,angle=auto_correct_image_orientation(image,mask)
        return image
    #-------------------------------------------------------------------------------------------------------------------------
    # extractions
    #-------------------------------------------------------------------------------------------------------------------------
    def get_basic_info(self,box_dict,crops):
        basic={}
        # english ocr
        eng_keys=["nid","dob"]
        ## dob
        dob    = box_dict[eng_keys[1]]
        dob_crops=[crops[i] for i in dob]
        ## id
        idx    = box_dict[eng_keys[0]]
        idx_crops=[crops[i] for i in idx]
        
        en_crops=dob_crops+idx_crops
        en_text = self.base.ocr(en_crops,det=False,cls=False)
        en_text = [i[0] for i in en_text]
        dob="".join(en_text[:len(dob_crops)])
        en_text=en_text[len(dob_crops):]
        idx="".join(en_text)

        
        basic["nid"]=processNID(idx) 
        basic["dob"]=processDob(dob)
        basic["success"]="true" 

        if basic["nid"] is None:
            basic["nid"]="failed"
            basic["success"]="false"
                
        if basic["dob"] is None:
            basic["dob"]="failed"
            basic["success"]="false"
        return basic 
    

            
    def __call__(self,img_path):
        # -----------------------start-----------------------
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        src=np.copy(img)
        clss=[ 'dob', 'nid']
        # mask
        mask=self.det.detect(img,self.base,ret_mask=True)
        img=self.execute_rotation_fix(img,mask)
        # check yolo
        img,locs=self.loc(img,clss)
        if img is None:
            return "error"
        else:
            # text detection
            boxes,crops=self.det.detect(img,self.base)
            # sorted box dictionary
            box_dict=self.process_boxes(boxes,locs,clss)    
            data=self.get_basic_info(box_dict,crops)
            return data 
            
