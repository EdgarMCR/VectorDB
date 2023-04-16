import azure.functions as func
import logging

APP_JSON = "application/json"

app = func.FunctionApp()

@app.function_name(name="Diagnostic")
@app.route(route="diagnostic")
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    return func.HttpResponse('{"status": "ok"}', mimetype=APP_JSON, status_code=200)