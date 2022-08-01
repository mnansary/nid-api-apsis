from __future__ import print_function
from scipy.misc import face
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import pathlib
from datetime import datetime
from flask import Flask,request,jsonify
from flask_restful import Resource, Api, reqparse
from werkzeug.utils import secure_filename
from time import time
from pprint import pprint
# models
from coreLib.ocr import OCR
# initialize ocr
# initialize ocr
ocr=OCR()


app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('name')


def handle_cardface(face):
    if face is None:
        return "front"
    elif face=="back":
        return "back"
    elif face=="front":
        return "front"
    else:
        return "invalid" 

def handle_includes(includes):
    # none case default
    if includes is None:
        return False 
    # single case
    elif "," not in includes:
        if includes=="bangla":
            return True
        else:
            return "invalid"
    else:
        return "invalid"
            
def handle_execs(executes):
    # none case default
    if executes is None:
        return True 
    # single case
    elif "," not in executes:
        if executes=="rotation-fix":
            return True
        else:
            return "invalid"
    else:
        return "invalid"
            
def consttruct_error(msg,etype,msg_code,details,suggestion=""):
    exec_error={"code":msg_code,
           "type":etype,
           "message":msg,
           "details":details,
           "suggestion":suggestion}
    return exec_error


def update_log(logs):
    with open("logs.log","a+") as log:
        log.write("..............................................\n")
        for k in logs.keys():
            log.write(f"{k}:\t{logs[k]}\n")
        log.write("----------------------------------------------\n")



class GetFile(Resource):
    def get(self):
        return {'test': 'this is test'}	
        
    def post(self):
        try:
            # container
            logs={}
            logs["req-time"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            
            req_start=time()
            # handle card face
            face=handle_cardface(request.args.get("cardface"))
            if face =="invalid":
                logs["error"]=f'received cardface:{request.args.get("cardface")}'
                update_log(logs)
                return jsonify({"error": consttruct_error("wrong cardface parameter",
                                                            "INVALID_PARAMETER",
                                                            "400",
                                                            f'received cardface:{request.args.get("cardface")}',
                                                            "valid cardface:front,back")}) 

            # handle includes
            ret_bangla=handle_includes(request.args.get("includes"))
            if ret_bangla =="invalid":
                logs["error"]=f'received includes:{request.args.get("includes")}'
                update_log(logs)
                return jsonify({"error":consttruct_error("wrong includes parameter",
                                                            "INVALID_PARAMETER",
                                                            "400",
                                                            f'received includes:{request.args.get("includes")}',
                                                            "valid includes: bangla")})

            # handle executes
            exec_rot=handle_execs(request.args.get("executes"))
            if exec_rot=="invalid":
                logs["error"]=f'received executes:{request.args.get("executes")}'
                update_log(logs)
                return jsonify({"error":consttruct_error("wrong executes parameter",
                                                            "INVALID_PARAMETER",
                                                            "400",
                                                            f'received executes:{request.args.get("executes")}',
                                                            "valid executes:rotation-fix") })
                
            try:
                save_start=time()
                basepath = os.path.dirname(__file__)
                if "file" in request.files:
                    # Get the file from post request
                    f = request.files['file']
                    file_path = os.path.join(basepath,"tests",secure_filename(f.filename))
                    # save file
                    file_ext=pathlib.Path(file_path).suffix
                    if file_ext not in [".jpg",".png",".jpeg"]:
                        logs["error"]=f"received file-extension:{file_ext}"
                        update_log(logs)
                        return jsonify({"error":consttruct_error("image format not valid.",
                                                                "INVALID_IMAGE","400",
                                                                f"received file-extension:{file_ext}",
                                                                "Please send .png image files")})
                    
                    f.save(file_path)
                    logs["file-name"]=secure_filename(f.filename)    
            except Exception as ef:
                logs["error"]="nidimage not received"
                update_log(logs)
                return jsonify({"error":consttruct_error("nidimage not received",
                                                            "INVALID_PARAMETER",
                                                            "400",
                                                            "",
                                                            "Please send image as form data")})
                
            
            
            logs["file-save-time"]=round(time()-save_start,2)
            logs["card-face"]=face
            logs["params"]={"bangla":ret_bangla,"rotation-fix":exec_rot}
            
            try:
                img=cv2.imread(file_path)
            except Exception as er:
                logs["error"]="image not readable."
                update_log(logs)
                return jsonify({"error":consttruct_error("image not readable.",
                                                            "INVALID_IMAGE",
                                                            "400",
                                                            "",
                                                            "Please send uncorrupted image files")})
            
                
            
            proc_start=time()
            ocr_out=ocr(file_path,face,ret_bangla,exec_rot)
            logs["ocr-processing-time"]=round(time()-proc_start,2)
            
            if ocr_out =="loc-error":
                logs["error"]="key fields cant be clearly located"
                update_log(logs)
                return jsonify({"error":consttruct_error("image is problematic",
                                                            "INVALID_IMAGE","400",
                                                            "key fields cant be clearly located",
                                                            "please try again with a clear nid image")})
            elif "coverage-error#" in ocr_out:
                logs["error"]=f"Text region coverage:{ocr_out.replace('coverage-error#','')}, which is lower than 30%"
                update_log(logs)
                return jsonify({"error":consttruct_error("image is problematic",
                                                            "INVALID_IMAGE","400",
                                                            f"Text region coverage:{ocr_out.replace('coverage-error#','')}, which is lower than 30%",
                                                            "please try again with a clear nid image")})

            elif ocr_out=="text-region-missing":
                logs["error"]="No text region found. Probably not an nid image."
                update_log(logs)
                return jsonify({"error":consttruct_error("image is problematic",
                                                            "INVALID_IMAGE","400",
                                                            "No text region found. Probably not an nid image.",
                                                            "please try again with a clear nid image")})

            elif ocr_out=="no-fields":
                logs["error"]="No key-fields are detected."
                update_log(logs)
                return jsonify({"error":consttruct_error("image is problematic",
                                                            "INVALID_IMAGE","400",
                                                            "No key-fields are detected.",
                                                            "please try again with a clear nid image")})

            elif ocr_out=="addr-not-located":
                logs["error"]="address cant be located properly"
                update_log(logs)
                return jsonify({"error":consttruct_error("image is problematic",
                                                            "INVALID_IMAGE","400",
                                                            "address cant be located properly",
                                                            "please try again with a clear nid image")})
            
            
            data={}
            data["data"]=ocr_out
            logs["req-handling-time"]=round(time()-req_start,2)
            
            update_log(logs)
            os.remove(file_path)

            res={"nid":data["data"]["nid-basic-info"]["nid"],
                    "dob":data["data"]["nid-basic-info"]["dob"]}
            return jsonify(res)
    
        except Exception as e:
             return jsonify({"error":consttruct_error("","INTERNAL_SERVER_ERROR","500","","please try again with a different image")})
    
        


api.add_resource(GetFile, '/nid')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2088)