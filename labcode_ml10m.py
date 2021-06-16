import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler_time import LoadData, negSamp, transToLsts, transpose, prepareGlobalData, sampleLargeGraph
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

class Recommender:
	def __init__(self, sess, datas):
		self.sess = sess
		self.trnMats, self.iiMats, self.tstInt, self.label, self.tstUsrs, args.intTypes, self.maxTime = datas
		prepareGlobalData(self.trnMats, self.label, self.iiMats)
		args.user, args.item = self.trnMats[0].shape
		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train'+met] = list()
			self.metrics['Test'+met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss'])
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Varaibles Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % 3 == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % 5 == 0:
				self.saveHistory()
			print()
		reses = self.smallTestEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def GAT(self, srcEmbeds, tgtEmbeds, tgtNodes, maxNum, Qs, Ks, Vs):
		QWeight = tf.nn.softmax(NNs.defineRandomNameParam([args.memosize, 1, 1], reg=True), axis=1)
		KWeight = tf.nn.softmax(NNs.defineRandomNameParam([args.memosize, 1, 1], reg=True), axis=1)
		VWeight = tf.nn.softmax(NNs.defineRandomNameParam([args.memosize, 1, 1], reg=True), axis=1)
		Q = tf.reduce_sum(Qs * QWeight, axis=0)
		K = tf.reduce_sum(Ks * KWeight, axis=0)
		V = tf.reduce_sum(Vs * VWeight, axis=0)

		q = tf.reshape(tgtEmbeds @ Q, [-1, args.att_head, args.latdim//args.att_head])
		k = tf.reshape(srcEmbeds @ K, [-1, args.att_head, args.latdim//args.att_head])
		v = tf.reshape(srcEmbeds @ V, [-1, args.att_head, args.latdim//args.att_head])
		logits = tf.math.exp(tf.reduce_sum(q * k, axis=-1, keepdims=True) / tf.sqrt(args.latdim/args.att_head))
		attNorm = tf.nn.embedding_lookup(tf.math.segment_sum(logits, tgtNodes), tgtNodes) + 1e-6
		att = logits / attNorm
		padAttval = tf.pad(att * v, [[0, 1], [0, 0], [0, 0]])
		padTgtNodes = tf.concat([tgtNodes, tf.reshape(maxNum-1, [1])], axis=-1)
		attval = tf.reshape(tf.math.segment_sum(padAttval, padTgtNodes), [-1, args.latdim])
		return attval

	def messagePropagate(self, srclats, tgtlats, mats, maxNum, wTime=True):
		unAct = []
		lats1 = []
		paramId = 'dfltP%d' % NNs.getParamId()
		Qs = NNs.defineRandomNameParam([args.memosize, args.latdim, args.latdim], reg=True)
		Ks = NNs.defineRandomNameParam([args.memosize, args.latdim, args.latdim], reg=True)
		Vs = NNs.defineRandomNameParam([args.memosize, args.latdim, args.latdim], reg=True)
		for mat in mats:
			timeEmbed = FC(self.timeEmbed, args.latdim, reg=True)
			srcNodes = tf.squeeze(tf.slice(mat.indices, [0, 1], [-1, 1]))
			tgtNodes = tf.squeeze(tf.slice(mat.indices, [0, 0], [-1, 1]))
			edgeVals = mat.values
			srcEmbeds = (tf.nn.embedding_lookup(srclats, srcNodes) + (tf.nn.embedding_lookup(timeEmbed, edgeVals) if wTime else 0))
			tgtEmbeds = tf.nn.embedding_lookup(tgtlats, tgtNodes)

			newTgtEmbeds = self.GAT(srcEmbeds, tgtEmbeds, tgtNodes, maxNum, Qs, Ks, Vs)

			unAct.append(newTgtEmbeds)
			lats1.append(Activate(newTgtEmbeds, self.actFunc))

		lats2 = NNs.lightSelfAttention(lats1, number=len(mats), inpDim=args.latdim, numHeads=args.att_head)

		# aggregation gate
		globalQuery = Activate(tf.add_n(unAct), self.actFunc)
		weights = []
		paramId = 'dfltP%d' % NNs.getParamId()
		for lat in lats2:
			temlat = FC(tf.concat([lat, globalQuery], axis=-1) , args.latdim//2, useBias=False, reg=False, activation=self.actFunc, name=paramId+'_1', reuse=True)
			weight = FC(temlat, 1, useBias=False, reg=False, name=paramId+'_2', reuse=True)
			weights.append(weight)
		stkWeight = tf.concat(weights, axis=1)
		sftWeight = tf.reshape(tf.nn.softmax(stkWeight, axis=1), [-1, len(mats), 1]) * 8
		stkLat = tf.stack(lats2, axis=1)
		lat = tf.reshape(tf.reduce_sum(sftWeight * stkLat, axis=1), [-1, args.latdim])
		return lat

	def makeTimeEmbed(self):
		divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
		pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
		sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim*2]) / 4.0
		return timeEmbed

	def ours(self):
		all_uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
		all_iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)
		uEmbed0 = tf.nn.embedding_lookup(all_uEmbed0, self.all_usrs)
		iEmbed0 = tf.nn.embedding_lookup(all_iEmbed0, self.all_itms)
		self.timeEmbed = tf.Variable(initial_value=self.makeTimeEmbed(), shape=[self.maxTime, args.latdim*2], name='timeEmbed')
		NNs.addReg('timeEmbed', self.timeEmbed)
		ulats = [uEmbed0]
		ilats = [iEmbed0]
		for i in range(args.gnn_layer):
			ulat = self.messagePropagate(ilats[-1], ulats[-1], self.adjs, self.usrNum)
			ilat1 = self.messagePropagate(ulats[-1], ilats[-1], self.tpAdjs, self.itmNum)
			ilat2 = self.messagePropagate(ilats[-1], ilats[-1], self.iiAdjs, self.itmNum, wTime=False)
			ilat = args.iiweight * ilat2 + (1.0 - args.iiweight) * ilat1
			ulats.append(ulat + ulats[-1])
			ilats.append(ilat + ilats[-1])

		UEmbedPred = NNs.defineParam('UEmbedPred', shape=[args.user, args.latdim], dtype=tf.float32, reg=False)
		IEmbedPred = NNs.defineParam('IEmbedPred', shape=[args.item, args.latdim], dtype=tf.float32, reg=False)
		ulats[0] = tf.nn.embedding_lookup(UEmbedPred, self.all_usrs)
		ilats[0] = tf.nn.embedding_lookup(IEmbedPred, self.all_itms)

		ulat = tf.add_n(ulats)
		ilat = tf.add_n(ilats)
		pckULat = tf.nn.embedding_lookup(ulat, self.uids)
		pckILat = tf.nn.embedding_lookup(ilat, self.iids)

		predLat = pckULat * pckILat * args.mult

		for i in range(args.deep_layer):
			predLat = FC(predLat, args.latdim, reg=True, useBias=True, activation=self.actFunc) + predLat
		pred = tf.squeeze(FC(predLat, 1, reg=True, useBias=True))
		return pred

	def prepareModel(self):
		self.keepRate = tf.placeholder(name='keepRate', dtype=tf.float32, shape=[])
		self.actFunc = 'twoWayLeakyRelu6'
		self.adjs = []
		self.tpAdjs = []
		self.iiAdjs = []
		for i in range(args.intTypes):
			self.adjs.append(tf.sparse_placeholder(dtype=tf.int32))
			self.tpAdjs.append(tf.sparse_placeholder(dtype=tf.int32))
		for i in range(len(self.iiMats)):
			self.iiAdjs.append(tf.sparse_placeholder(dtype=tf.int32))

		self.all_usrs = tf.placeholder(name='all_usrs', dtype=tf.int32, shape=[None])
		self.all_itms = tf.placeholder(name='all_itms', dtype=tf.int32, shape=[None])
		self.usrNum = tf.placeholder(name='usrNum', dtype=tf.int64, shape=[])
		self.itmNum = tf.placeholder(name='itmNum', dtype=tf.int64, shape=[])
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])

		self.pred = self.ours()
		sampNum = tf.shape(self.iids)[0] // 2
		posPred = tf.slice(self.pred, [0], [sampNum])
		negPred = tf.slice(self.pred, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batchIds, itmnum, label):
		preSamp = list(np.random.permutation(itmnum))
		temLabel = label[batchIds].toarray()
		batch = len(batchIds)
		temlen = batch * 2 * args.sampNum
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			negset = negSamp(temLabel[i], preSamp)
			poslocs = np.random.choice(posset, args.sampNum)
			neglocs = np.random.choice(negset, args.sampNum)
			for j in range(args.sampNum):
				uIntLoc[cur] = uIntLoc[cur+temlen//2] = batchIds[i]
				iIntLoc[cur] = poslocs[j]
				iIntLoc[cur+temlen//2] = neglocs[j]
				cur += 1
		return uIntLoc, iIntLoc

	def trainEpoch(self):
		num = args.user
		allIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(allIds)

		bigSteps = int(np.ceil(num / args.graphSampleN))
		glb_i = 0
		glb_step = int(np.ceil(num / args.batch))
		for s in range(bigSteps):
			bigSt = s * args.graphSampleN
			bigEd = min((s+1) * args.graphSampleN, num)
			sfIds = allIds[bigSt: bigEd]

			steps = int(np.ceil((bigEd - bigSt) / args.batch))

			pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms = sampleLargeGraph(sfIds)
			pckLabel = transpose(transpose(self.label[usrs])[itms])
			usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
			sfIds = list(map(lambda x: usrIdMap[x], sfIds))
			feeddict = {self.all_usrs: usrs, self.all_itms: itms, self.usrNum: len(usrs), self.itmNum: len(itms)}
			for i in range(args.intTypes):
				feeddict[self.adjs[i]] = transToLsts(pckAdjs[i])
				feeddict[self.tpAdjs[i]] = transToLsts(pckTpAdjs[i])
			for i in range(len(pckIiAdjs)):
				feeddict[self.iiAdjs[i]] = transToLsts(pckIiAdjs[i])

			for i in range(steps):
				st = i * args.batch
				ed = min((i+1) * args.batch, bigEd - bigSt)
				batIds = sfIds[st: ed]

				uLocs, iLocs = self.sampleTrainBatch(batIds, pckAdjs[0].shape[1], pckLabel)

				target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
				feeddict[self.uids] = uLocs
				feeddict[self.iids] = iLocs
				res = self.sess.run(target, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

				preLoss, regLoss, loss = res[1:]

				epochLoss += loss
				epochPreLoss += preLoss
				glb_i += 1
				log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (glb_i, glb_step, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / glb_step
		ret['preLoss'] = epochPreLoss / glb_step
		return ret

	def sampleTestBatch(self, batchIds, label, tstInt):
		batch = len(batchIds)
		temTst = tstInt[batchIds]
		temLabel = label[batchIds].toarray()
		temlen = batch * 100
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			rdnNegSet = np.random.permutation(negset)[:99]
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uIntLoc[cur] = batchIds[i]
				iIntLoc[cur] = locset[j]
				cur += 1
		return uIntLoc, iIntLoc, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		ids = self.tstUsrs
		num = len(ids)
		tstBat = np.maximum(1, args.batch * args.sampNum // 100)
		steps = int(np.ceil(num / tstBat))

		posItms = self.tstInt[ids]
		pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms = sampleLargeGraph(ids, list(set(posItms)))
		pckLabel = transpose(transpose(self.label[usrs])[itms])
		usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
		itmIdMap = dict(map(lambda x: (itms[x], x), range(len(itms))))
		ids = list(map(lambda x: usrIdMap[x], ids))
		itmMapping = (lambda x: None if (x is None) else itmIdMap[x])
		pckTstInt = np.array(list(map(lambda x: itmMapping(self.tstInt[usrs[x]]), range(len(usrs)))))
		feeddict = {self.all_usrs: usrs, self.all_itms: itms, self.usrNum: len(usrs), self.itmNum: len(itms)}
		for i in range(args.intTypes):
			feeddict[self.adjs[i]] = transToLsts(pckAdjs[i])
			feeddict[self.tpAdjs[i]] = transToLsts(pckTpAdjs[i])
		for i in range(len(pckIiAdjs)):
			feeddict[self.iiAdjs[i]] = transToLsts(pckIiAdjs[i])

		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, pckLabel, pckTstInt)
			feeddict[self.uids] = uLocs
			feeddict[self.iids] = iLocs
			preds = self.sess.run(self.pred, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Steps %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def smallTestEpoch(self):
		epochHit, epochNdcg = [0] * 2
		allIds = self.tstUsrs
		num = len(allIds)
		tstBat = np.maximum(1, args.batch * args.sampNum // 100)

		divSize = args.divSize#args.graphSampleN
		bigSteps = int(np.ceil(num / divSize))
		glb_i = 0
		glb_step = int(np.ceil(num / tstBat))
		for s in range(bigSteps):
			bigSt = s * divSize
			bigEd = min((s+1) * divSize, num)
			ids = allIds[bigSt: bigEd]

			steps = int(np.ceil((bigEd - bigSt) / tstBat))

			posItms = self.tstInt[ids]
			pckAdjs, pckTpAdjs, pckIiAdjs, usrs, itms = sampleLargeGraph(ids, list(set(posItms)))
			pckLabel = transpose(transpose(self.label[usrs])[itms])
			usrIdMap = dict(map(lambda x: (usrs[x], x), range(len(usrs))))
			itmIdMap = dict(map(lambda x: (itms[x], x), range(len(itms))))
			ids = list(map(lambda x: usrIdMap[x], ids))
			itmMapping = (lambda x: None if (x is None or x not in itmIdMap) else itmIdMap[x])
			pckTstInt = np.array(list(map(lambda x: itmMapping(self.tstInt[usrs[x]]), range(len(usrs)))))
			feeddict = {self.all_usrs: usrs, self.all_itms: itms, self.usrNum: len(usrs), self.itmNum: len(itms)}
			for i in range(args.intTypes):
				feeddict[self.adjs[i]] = transToLsts(pckAdjs[i])
				feeddict[self.tpAdjs[i]] = transToLsts(pckTpAdjs[i])
			for i in range(len(pckIiAdjs)):
				feeddict[self.iiAdjs[i]] = transToLsts(pckIiAdjs[i])

			for i in range(steps):
				st = i * tstBat
				ed = min((i+1) * tstBat, bigEd - bigSt)
				batIds = ids[st: ed]
				uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, pckLabel, pckTstInt)
				feeddict[self.uids] = uLocs
				feeddict[self.iids] = iLocs
				preds = self.sess.run(self.pred, feed_dict=feeddict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
				hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
				epochHit += hit
				epochNdcg += ndcg
				glb_i += 1
				log('Steps %d/%d: hit = %d, ndcg = %d          ' % (glb_i, glb_step, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	datas = LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, datas)
		recom.run()