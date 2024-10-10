from flask import Flask, jsonify
from flask import Flask, request
import util_model
app = Flask(__name__)
import subprocess
import init_model

# List of Python scripts to run
scripts = ['util_model.py', 'init_model.py']
for script in scripts:
    result = subprocess.run(['python3', script])
    if result.returncode != 0:
        print(f"Script {script} failed with error.")
    else:
        print(f"Script {script} ran successfully.")

print("All scripts have finished running.")

@app.route('/api/fin_model/analyze', methods=['POST'])
def analyze():
    objModel = init_model.InitModel()

    args = request.args
    #texts = args.get('text', '')
    texts = request.form.get('texts')
    json_req = request.json
    print(f'text : {texts}')
    
    text_print = util_model.print_pred_sentiment([json_req['texts']], objModel.loaded_tokenizer, objModel.loaded_model, objModel.X)
    return jsonify(
        {
            "sentiment: ": f"{text_print}" 
            }
        )

if __name__ == '__main__':
    app.run(debug=True)