from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
from time import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Models
from coreLib.ocr import OCR

# Initialize OCR
basepath = os.path.dirname(__file__)
ocr = OCR(yolo_onnx=os.path.join(basepath, "weights/yolo.onnx"))

app = Flask(__name__)
api = Api(app)

# JWT Configuration
app.config['JWT_SECRET_KEY'] = '41a5f3a411ab413f82325e7deff027bd'  # Change this to your secret key
jwt = JWTManager(app)

# Rate Limiting Configuration
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["14400 per day", "600 per hour"]
)

parser = reqparse.RequestParser()
parser.add_argument('name')

def consttruct_error(msg, etype, msg_code, details, suggestion=""):
    exec_error = {"code": msg_code,
                  "type": etype,
                  "message": msg,
                  "details": details,
                  "suggestion": suggestion}
    return exec_error

def update_log(logs):
    with open("logs.log", "a+") as log:
        for k in logs.keys():
            log.write(f"{k}:\t{logs[k]}\t")
        log.write("\n")

@jwt.unauthorized_loader
def custom_unauthorized_response(callback):
    response = jsonify({
        'error': consttruct_error("Missing or invalid Authorization header",
                                  "UNAUTHORIZED",
                                  "401",
                                  "Authorization header is missing or invalid",
                                  "Please provide a valid JWT token"),
        'success': "false"
    })
    response.status_code = 401
    return response

class GetFile(Resource):
    @jwt_required()
    @limiter.limit("10 per minute")
    def get(self):
        return {'test': 'this is test'}

    @jwt_required()
    @limiter.limit("10 per minute")
    def post(self):
        try:
            logs = {}
            time_stamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
            start = time()
            try:
                basepath = os.path.dirname(__file__)
                if "file" in request.files:
                    f = request.files['file']
                    file_path = os.path.join(basepath, "tests", f"upload_{time_stamp}.jpg")
                    f.save(file_path)
                    logs["file-name"] = f"upload_{time_stamp}.jpg"
            except Exception as ef:
                logs["error"] = "nidimage not received"
                update_log(logs)
                return jsonify({"error": consttruct_error("nidimage not received",
                                                          "INVALID_PARAMETER",
                                                          "400",
                                                          "",
                                                          "Please send image as form data"), "success": "false"})
            
            try:
                img = cv2.imread(file_path)
            except Exception as er:
                logs["error"] = "image not readable."
                update_log(logs)
                return jsonify({"error": consttruct_error("image not readable.",
                                                          "INVALID_IMAGE",
                                                          "400",
                                                          "",
                                                          "Please send uncorrupted image files"), "success": "false"})
            
            ocr_out = ocr(file_path)
            if ocr_out == "error":
                logs["error"] = "key fields cant be clearly located"
                update_log(logs)
                return jsonify({"error": consttruct_error("image is problematic",
                                                          "INVALID_IMAGE", "400",
                                                          "key fields cant be clearly located",
                                                          "please try again with a clear nid image"), "success": "false"})
            
            logs["res"] = ocr_out
            end = time()
            logs["time"] = round(end - start, 2)
            update_log(logs)
            return jsonify(ocr_out)

        except Exception as e:
            return jsonify({"error": consttruct_error("", "NETWORK/SERVER ERROR", "500", "", ""), "success": "false"})

api.add_resource(GetFile, '/nid')

if __name__ == '__main__':
    # Manually create a JWT token for testing
    with app.app_context():
        token = create_access_token(identity='test_user_batb')
        print(f"Use this JWT token for testing: {token}")

    app.run(debug=False, host='0.0.0.0', port=3040)