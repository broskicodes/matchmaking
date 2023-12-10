from flask import Flask, jsonify, send_from_directory, request
import interests_map

app = Flask(__name__)

@app.route('/graphic', methods=['GET'])
def send_visualization():
    interests_map.create_visualization(interests_map.load_data())
    
    return send_from_directory('data', 'visualization.png', mimetype='image/png')
  
@app.route('/interests', methods=['GET'])
def send_ppl_csv():
    return send_from_directory('data', 'ppl.csv', mimetype='text/csv')
  
@app.route('/interests', methods=['POST'])
def add_interests():
    json_body = request.get_json()
    
    name, interest, sp_id = json_body['name'], json_body['interest'], json_body['id']    
    interests_map.add_interests([(name, interest, sp_id)])
    
    return jsonify({'success': True})

if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0')
    app.run()
