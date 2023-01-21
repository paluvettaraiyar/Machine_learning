from flask import Flask
from flask_cors import CORS,cross_origin

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return "hai"


@app.route('/hello',methods=['GET','POST'])
@cross_origin()
def helloworld():
    return "helloworld"

if __name__ == "__main__":
    app.run(debug=True)
