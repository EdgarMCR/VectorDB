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


@app.function_name(name="Diagnostic")
@app.route(route="diagnostic")
def test_function(req: func.HttpRequest, methods=[func.HttpMethod.GET]) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    body = req.get_json()

    if not 'max-distance' in body:
        return func.HttpResponse('{"msg": "No key `max-distance` in JSON"}', mimetype=APP_JSON, status_code=400)
    max_distance = body['max-distance']
    try:
        max_distance = float(max_distance)
    except Exception:
        return func.HttpResponse('{"msg": "`max-distance` not a float"}', mimetype=APP_JSON, status_code=400)

    if not 'vector' in body:
        return func.HttpResponse('{"msg": "No key `vector` in JSON"}', mimetype=APP_JSON, status_code=400)

    vector = body['vector']
    if len(vector) != 512:
        return func.HttpResponse('{"msg": "Vector length is not 512"}', mimetype=APP_JSON, status_code=400)

    for x in vector:
        if not isinstance(x, float):
            return func.HttpResponse('{"msg": "Vector should only contain floats"}', mimetype=APP_JSON, status_code=400)

    results = mv.get_all_matches_within_distance(vector, max_distance)

    return func.HttpResponse(json.dumps(results), mimetype=APP_JSON, status_code=200)
