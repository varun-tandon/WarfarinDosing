class Config:
    def __init__(self, model, seed):
        self.model_type = model
        seed_str = "seed=" + str(seed)
        self.output_path = "results/{}-{}/".format(
            self.model_type, seed_str
        )
        self.accuracy_output = self.output_path + "accuracy.npy"
        self.regret_output = self.output_path + "regret.npy"
