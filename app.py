from flask import Flask, render_template, request
import utils
app = Flask(__name__)



@app.route('/',methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
      result = request.form
      model_path_d2v_2 = "models/d2v_Jun-15-2022"

      sim = utils.similarity_finder(result['text1'], result['text2'], model_path_d2v_2)
      print(sim)
      return render_template("index.html",result = sim)
      
    else:
        return render_template("index.html")

if __name__ == '__main__':
   app.run(debug = True)