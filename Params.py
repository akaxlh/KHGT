import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=32, type=int, help='batch size')
	parser.add_argument('--reg', default=1e-2, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=120, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=16, type=int, help='embedding size')
	parser.add_argument('--memosize', default=2, type=int, help='memory size')
	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
	parser.add_argument('--target', default='buy', type=str, help='target behavior to predict on')
	parser.add_argument('--deep_layer', default=0, type=int, help='number of deep layers to make the final prediction')
	parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')
	parser.add_argument('--keepRate', default=0.7, type=float, help='rate for dropout')
	parser.add_argument('--iiweight', default=0.3, type=float, help='weight for ii')
	parser.add_argument('--slot', default=5, type=int, help='length of time slots')
	parser.add_argument('--graphSampleN', default=25000, type=int, help='use 25000 for training and 200000 for testing, empirically')
	parser.add_argument('--divSize', default=50, type=int, help='div size for smallTestEpoch')
	return parser.parse_args()
args = parse_args()
# args.user = 147894
# args.item = 99037
# ML10M
# args.user = 67788
# args.item = 8704
# yelp
# args.user = 19800
# args.item = 22734


args.decay_step = args.trnNum//args.batch
