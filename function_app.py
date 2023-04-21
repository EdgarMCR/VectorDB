import json

import azure.functions as func
import logging

import src.match_vector as mv


APP_JSON = "application/json"

app = func.FunctionApp()

@app.function_name(name="Diagnostic")
@app.route(route="diagnostic")
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    return func.HttpResponse('{"status": "ok"}', mimetype=APP_JSON, status_code=200)


def get_vector_and_max_distance(body: dict):
    err_msg, max_distance, min_confidence, vector = '', None, None, None
    if not 'max-distance' in body:
        return "No key `max-distance` in JSON", max_distance, min_confidence, vector

    max_distance = body['max-distance']
    try:
        max_distance = float(max_distance)
    except Exception:
        return "`max-distance` not a float", max_distance, min_confidence, vector
    
    if not 'min-confidence' in body:
        return "No key `min-confidence` in JSON", max_distance, min_confidence, vector
    
    min_confidence = body['min-confidence']
    try:
        min_confidence = float(min_confidence)
    except Exception:
        return "`min-confidence` not a float", min_confidence, min_confidence, vector
    
    if not 'vector' in body:
        return "No key `vector` in JSON", max_distance, min_confidence, vector

    vector = body['vector']
    if len(vector) != 512:
        return "Vector length is not 512", max_distance, min_confidence, vector

    for x in vector:
        if not isinstance(x, float):
            return "Vector should only contain floats", max_distance, min_confidence, vector
    return err_msg, max_distance, min_confidence, vector


@app.function_name(name="Match")
@app.route(route="match", methods=[func.HttpMethod.POST])
def match(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse('{"msg": "HTTP request does not contain valid JSON data"}', mimetype=APP_JSON, status_code=400)
    
    err_msg, max_distance, min_confidence, vector = get_vector_and_max_distance(body)

    if err_msg:
        return func.HttpResponse('{"msg": "' + err_msg + '"}', mimetype=APP_JSON, status_code=400)

    results = mv.get_all_matches_within_distance(vector, max_distance, min_confidence, False)

    return func.HttpResponse(json.dumps(results), mimetype=APP_JSON, status_code=200)


@app.function_name(name="MatchWithVectors")
@app.route(route="match-with-vectors", methods=[func.HttpMethod.POST])
def match_with_vector(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse('{"msg": "HTTP request does not contain valid JSON data"}', mimetype=APP_JSON, status_code=400)

    err_msg, max_distance, min_confidence, vector = get_vector_and_max_distance(body)

    if err_msg:
        return func.HttpResponse('{"msg": "' + err_msg + '"}', mimetype=APP_JSON, status_code=400)

    results = mv.get_all_matches_within_distance(vector, max_distance, min_confidence, True)

    return func.HttpResponse(json.dumps(results), mimetype=APP_JSON, status_code=200)
