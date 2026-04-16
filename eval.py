from comet import download_model, load_from_checkpoint

class Evaluator:

    def __init__(self):

        model_path = download_model("Unbabel/XCOMET-XL")
        self.model = load_from_checkpoint(model_path)

    def evaluate(self, data, batch_size):
        """
        Evaluates model outputs using XCOMET. 
        Expects data as a list of dicts containing "src", "mt", and "ref" keys.
        """

        model_output = self.model.predict(data, batch_size=batch_size, gpus=1)
        
        scores = {
                "segment": model_output.scores,
                "system": model_output.system_score,
                "error": model_output.metadata.error_spans
                }

        return scores 
