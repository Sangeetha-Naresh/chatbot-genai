from flask import Flask, render_template, request, jsonify
from generative_response import generate_bot_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/bot-response", methods=["POST"])
def bot():
    # Get User Input
    input_text = request.json.get("user_bot_input_text")
   
    # Call the method to get bot response
    bot_res = generate_bot_response(input_text)

    response = {
        "bot_response": bot_res
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

