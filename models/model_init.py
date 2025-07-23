from models.directed.map import MAP
from models.directed.mapplus import MAPplus
from models.directed.mlp import MLP


class ModelZoo:
    def __init__(self, logger, args, num_nodes, feat_dim, output_dim, label, test_idx, task_level):
        super(ModelZoo, self).__init__()
        self.logger = logger
        self.args = args
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.task_level = task_level
        self.label = label
        self.test_idx = test_idx
        self.q = self.args.edge_q if self.task_level == "edge" else self.args.node_q
        self.log_model()

    def log_model(self):
        if self.args.model_name == "map":
            self.logger.info(f"model: {self.args.model_name}, q: {self.q}, hidden_dim: {self.args.hidden_dim}, dropout: {self.args.dropout}")
        elif self.args.model_name == "mapplus":
            self.logger.info(f"model: {self.args.model_name}, prop_steps: {self.args.prop_steps}, num_layers: {self.args.num_layers}, q: {self.q}, hidden_dim: {self.args.hidden_dim}, dropout: {self.args.dropout}")

    def model_init(self):
        if self.args.model_name == "map":
            model = MAP(q=self.q, feat_dim=self.feat_dim, hidden_dim=self.args.hidden_dim, output_dim=self.output_dim, 
                        label=self.label, test_idx=self.test_idx, dropout=self.args.dropout, task_level=self.task_level)
            
        elif self.args.model_name == "mapplus":
            model = MAPplus(prop_steps=self.args.prop_steps, num_layers=self.args.num_layers, use_att=self.args.use_att,
                            q=self.q, feat_dim=self.feat_dim, hidden_dim=self.args.hidden_dim, output_dim=self.output_dim,
                            label=self.label, test_idx=self.test_idx, dropout=self.args.dropout, task_level=self.task_level)

        elif self.args.model_name == "mlp":
            model = MLP(feat_dim=self.feat_dim, hidden_dim=self.args.hidden_dim, out_dim=self.output_dim)
            
        else:
            return NotImplementedError

        return model
