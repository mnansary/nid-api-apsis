from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import requests
from time import time
import pathlib
from datetime import datetime
from flask import Flask,request,jsonify,Response
from flask_restful import Resource, Api, reqparse
# models
from coreLib.ocr import OCR
# initialize ocr
# initialize ocr
basepath = os.path.dirname(__file__)
ocr=OCR(yolo_onnx=os.path.join(basepath,"weights/yolo.onnx"))


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('name')


            
def consttruct_error(msg,etype,msg_code,details,suggestion=""):
    exec_error={"code":msg_code,
           "type":etype,
           "message":msg,
           "details":details,
           "suggestion":suggestion}
    return exec_error


def update_log(logs):
    with open("logs.log","a+") as log:
        for k in logs.keys():
            log.write(f"{k}:\t{logs[k]}\t")
        log.write("\n")



class GetFile(Resource):
    def get(self):
        return {'test': 'this is test'}	
        
    def post(self):
        try:
            # container
            logs={}
            time_stamp=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            start=time()
            try:
                basepath = os.path.dirname(__file__)
                if "file" in request.files:
                    # Get the file from post request
                    f = request.files['file']
                    file_path = os.path.join(basepath,"tests",f"upload_{time_stamp}.jpg")
                    f.save(file_path)
                    logs["file-name"]=f"upload_{time_stamp}.jpg"    
            except Exception as ef:
                logs["error"]="nidimage not received"
                update_log(logs)
                return jsonify({"error":consttruct_error("nidimage not received",
                                                            "INVALID_PARAMETER",
                                                            "400",
                                                            "",
                                                            "Please send image as form data"),"success":"false"})
                
            
            
            
            try:
                img=cv2.imread(file_path)
            except Exception as er:
                logs["error"]="image not readable."
                update_log(logs)
                return jsonify({"error":consttruct_error("image not readable.",
                                                            "INVALID_IMAGE",
                                                            "400",
                                                            "",
                                                            "Please send uncorrupted image files"),"success":"false"})
            
                
            
            ocr_out=ocr(file_path)
            if ocr_out =="error":
                logs["error"]="key fields cant be clearly located"
                update_log(logs)
                return jsonify({"error":consttruct_error("image is problematic",
                                                        "INVALID_IMAGE","400",
                                                        "key fields cant be clearly located",
                                                        "please try again with a clear nid image"),"success":"false"})
            
            logs["res"]=ocr_out
            end=time()
            logs["time"]=round(end-start,2)
            update_log(logs)
            return jsonify(ocr_out)

        except Exception as e:
             return jsonify({"error":consttruct_error("","NETWORK/SERVER ERROR","500","",""),"success":"false"})




# Add resources to the API
api.add_resource(GetFile, '/nid')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=2088)