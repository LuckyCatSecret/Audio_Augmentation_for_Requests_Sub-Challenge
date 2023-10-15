import copy
import os
import sys
import time
from collections import OrderedDict, defaultdict

import torch

import numpy as np
from hyperopt import hp
import ray
import gorilla
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.suggest.hyperopt import HyperOptSearch
# from ray.tune.suggest import HyperOptSearch
from ray.tune import register_trainable, run_experiments
from tqdm import tqdm

# from archive import remove_deplicates, policy_decoder
# from FastAutoAugment.archive import remove_deplicates, policy_decoder
from .archive import remove_deplicates, policy_decoder
# from augmentations import augment_list
# from FastAutoAugment.augmentations import augment_wav_list,augment_spec_list,augment_cv_list
from .augmentations import augment_wav_list,augment_spec_list,augment_cv_list
# from common import get_logger, add_filehandler
# from FastAutoAugment.common import get_logger, add_filehandler
from .common import get_logger, add_filehandler
# from data import get_dataloaders
# from FastAutoAugment.data import get_dataloaders
# from FastAutoAugment.data import get_dataloaders
from .data import get_dataloaders
# from metrics import Accumulator
# from FastAutoAugment.metrics import Accumulator
from .metrics import Accumulator
# from networks import get_model, num_class
# from FastAutoAugment.networks import get_model, num_class
# from train import train_and_eval
# from FastAutoAugment.train import train_and_eval
# from .train import train_and_eval
from theconf import Config as C, ConfigArgumentParser

# from AMT_AA_models import build_model
from models import build_model
from .stats_map import calculate_stats

top1_valid_by_cv = defaultdict(lambda: list)


def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, 'step')

    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    # best_top1_acc = 0.
    best_top1_mAP = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        # best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])
        best_top1_mAP = max(best_top1_mAP, trial.last_result['mAP'])
    print('iter', self._iteration, 'top1_mAP=%.3f' % best_top1_mAP, cnts, end='\r')
    # print('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')
    return original(self)


patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)


logger = get_logger('Fast AutoAugment')


def _get_path(dataset, model, tag):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models/%s_%s_%s.model' % (dataset, model, tag))     # TODO


@ray.remote(num_gpus=1, max_calls=1)
def train_model(config, dataroot, augment, cv_ratio_test, cv_fold, save_path=None, skip_exist=False):
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment

    result = train_and_eval(None, dataroot, cv_ratio_test, cv_fold, save_path=save_path, only_eval=skip_exist)
    return C.get()['model']['type'], cv_fold, result


def eval_tta(config, augment, reporter):
    C.get()
    C.get().conf = config
    cv_ratio_test, cv_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']
    amt_conf=augment['amtconfig']

    # setup - provided augmentation rules
    C.get()['aug'] = policy_decoder(augment, augment['num_policy'], augment['num_op'])

    # eval
    # model = get_model(C.get()['model'], num_class(C.get()['dataset']))
    # model = build_model(C.get()['model'], is_pretrain=False)
    model = build_model(amt_conf, is_pretrain=False)
    # save_path=os.path.join('..',save_path)
    ckpt = torch.load(save_path, map_location=torch.device('cpu'))
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    loaders = []
    for _ in range(augment['num_policy']):  # TODO
    # for _ in range(1):  # TODO
        _, tl, validloader, tl2 = get_dataloaders(C.get()['data'],C.get()['dataset'],C.get()['batch'], augment['dataroot'], cv_ratio_test, split_idx=cv_fold)
        loaders.append(iter(validloader))
        del tl, tl2

    start_t = time.time()
    metrics = Accumulator()
    # loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_fn = torch.nn.BCEWithLogitsLoss()
    A_predictions = []
    A_targets = []
    losses = []
    try:
        while True:
            for loader in loaders:
                data, label = next(loader)
                data = data.cuda()
                label = label.cuda()

                data = data.unsqueeze(1)
                data = data.transpose(2, 3)

                pred = model(data)
                # audio_feature = model.module.forward_features(data).to('cpu').detach()

                audio_output = torch.sigmoid(pred)
                predictions = audio_output.to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(label)

                label = label.to(torch.device('cuda'))
                loss = loss_fn(audio_output, label)
                losses.append(loss.detach().cpu().numpy())

                # _, pred = pred.topk(1, 1, True, True)
                # pred = pred.t()
                # correct = pred.eq(label.view(1, -1).expand_as(pred)).detach().cpu().numpy()
                # corrects.append(correct)
                # del loss, correct, pred, data, label

            # losses = np.array(losses)
            # losses_min = np.min(losses, axis=0).squeeze()
    except StopIteration:
        pass

    losses = np.array(losses)
    audio_output = torch.cat(A_predictions).cuda().double()
    target = torch.cat(A_targets).cuda().long()
    stats = calculate_stats(audio_output.detach().cpu().numpy(), target.detach().cpu().numpy())
    # corrects = np.concatenate(corrects)
    # corrects_max = np.max(corrects, axis=0).squeeze()
    mAP = np.mean([stat['AP'] for stat in stats if stat['AP']!=-1])
    logger.info(mAP)
    metrics.add_dict({
        # 'minus_loss': -1 * np.sum(losses_min),
        'mean_loss': np.mean(losses),
        'mAP': mAP,
    })
    # del corrects, corrects_max
    del model
    # metrics = metrics / 'cnt'
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    reporter(minus_loss=metrics['mean_loss'], mAP=metrics['mAP'], elapsed_time=gpu_secs, done=True)
    return metrics['mAP']

### 封装为search policy
def search_policy(copied_c,path,amtconfig,aim_map):
    import json
    from pystopwatch2 import PyStopwatch
    w = PyStopwatch()
    
    # ray.init(local_mode=True)
    ray.init()
    num_result_per_cv = 20
    cv_num = 1
    
    ### start search
    w.start(tag='search')
    wav_ops = augment_wav_list()
    spec_ops = augment_spec_list()
    cv_ops = augment_cv_list()
    space = {}
    #spec
    for i in range(3):
        space['spec_policy_%d_%d' % (i, 0)] = hp.choice('spec_policy_%d_%d' % (i, 0), list(range(0, len(spec_ops))))
        space['spec_prob_%d_%d' % (i, 0)] = hp.uniform('spec_prob_%d_ %d' % (i, 0), 0.0, 1.0)
        space['spec_level_%d_%d' % (i, 0)] = hp.uniform('spec_level_%d_ %d' % (i, 0), 0.0, 1.0)
    for i in range(5):#args.num_policy=5
        for j in range(2):#args.num_op=2
            #wave
            space['wav_policy_%d_%d'%(i,j)]=hp.choice('wav_policy_%d_%d'%(i,j),list(range(0,len(wav_ops))))
            space['wav_prob_%d_%d'%(i,j)]=hp.uniform('wav_prob_%d_%d'%(i,j),0.0,1.0)
            #cv
            space['cv_policy_%d_%d' % (i, j)] = hp.choice('cv_policy_%d_%d' % (i, j), list(range(0, len(cv_ops))))
            space['cv_prob_%d_%d' % (i, j)] = hp.uniform('cv_prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['cv_level_%d_%d' % (i, j)] = hp.uniform('cv_level_%d_ %d' % (i, j), 0.0, 1.0)

    ### path 这里把finetuned好的模型传在了这里
    path=path
    # print(path)
    final_policy_set = []
    total_computation = 0
    reward_attr = 'mAP'      # top1_valid or minus_loss
    for _ in range(1):  # run multiple times.
        for cv_fold in range(cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (C.get()['dataset'], C.get()['model']['type'], cv_fold, 0.2)#args.cv_ratio=0.2
            print(name)
            register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_c), augs, rpt))
            algo = HyperOptSearch(space, max_concurrent=4*20, reward_attr=reward_attr)

            exp_config = {
                name: {
                    'run': name,
                    'num_samples': 20,#args.num_search=20
                    'resources_per_trial': {'gpu': 1},
                    'stop': {'training_iteration': 5},#args.num_policy=5
                    'config': {
                        'save_path': path,
                        'cv_ratio_test': 0.2, #args.cv_ratio=0.2
                        'cv_fold': cv_fold,
                        'num_op': 2, #args.num_op=2
                        'num_policy': 5, #args.num_policy=5
                        'amtconfig':amtconfig,
                        'dataroot':''
                    },
                }
            }
            results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, resume=False, raise_on_failed_trial=False)   #resume=args.resume='store_true'=False
            print()
            results = [x for x in results if x.last_result is not None]
            results = sorted(results, key=lambda x: x.last_result[reward_attr], reverse=True)

            # calculate computation usage
            for result in results:
                total_computation += result.last_result['elapsed_time']

            for result in results[:num_result_per_cv]:
                final_policy = policy_decoder(result.config, 5, 2)#args.num_policy=5,args.num_op=2
                logger.info('loss=%.12f mAP=%.4f %s' % (result.last_result['minus_loss'], result.last_result['mAP'], final_policy))
                if result.last_result['mAP']>=aim_map*0.7 and result.last_result['mAP']<=aim_map*1.5:
                    final_policy = remove_deplicates(final_policy)
                    final_policy_set.extend(final_policy)
    ray.shutdown()
    return final_policy_set

if __name__ == '__main__':
    import json
    from pystopwatch2 import PyStopwatch
    w = PyStopwatch()

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataroot', type=str, default='/data/private/pretrainedmodels', help='torchvision data folder')
    parser.add_argument('--until', type=int, default=5)
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--num-search', type=int, default=200)
    parser.add_argument('--cv-ratio', type=float, default=0.4)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--redis', type=str, default='gpu-cloud-vnode30.dakao.io:23655')
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay

    add_filehandler(logger, os.path.join('models', '%s_%s_cv%.1f.log' % (C.get()['dataset'], C.get()['model']['type'], args.cv_ratio)))
    logger.info('configuration...')
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))
    logger.info('initialize ray...')
    ### modify withou redis
    # ray.init(redis_address=args.redis)
    # ray.init()
    # ray.init(local_mode=True)

    ###
    # torch.set_default_tensor_type(torch.cuda.FloatStorage)

    num_result_per_cv = 50
    cv_num = 1
    # cv_num = 5
    copied_c = copy.deepcopy(C.get().conf)

    logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    logger.info('----- Train without Augmentations cv=%d ratio(test)=%.1f -----' % (cv_num, args.cv_ratio))
###         delete stage1           ###

    logger.info('----- Search Test-Time Augmentation Policies -----')
    w.start(tag='search')

    wav_ops = augment_wav_list()
    spec_ops = augment_spec_list()
    cv_ops = augment_cv_list()
    space = {}
    for i in range(3):
        space['spec_policy_%d_%d' % (i, 0)] = hp.choice('spec_policy_%d_%d' % (i, 0), list(range(0, len(spec_ops))))
        space['spec_prob_%d_%d' % (i, 0)] = hp.uniform('spec_prob_%d_ %d' % (i, 0), 0.0, 1.0)
        space['spec_level_%d_%d' % (i, 0)] = hp.uniform('spec_level_%d_ %d' % (i, 0), 0.0, 1.0)

    for i in range(args.num_policy):
        # space['spec_policy_%d_%d' % (i, 0)] = hp.choice('spec_policy_%d_%d' % (i, 0), list(range(0, len(spec_ops))))
        # space['spec_prob_%d_%d' % (i, 0)] = hp.uniform('spec_prob_%d_ %d' % (i, 0), 0.0, 1.0)
        # space['spec_level_%d_%d' % (i, 0)] = hp.uniform('spec_level_%d_ %d' % (i, 0), 0.0, 1.0)
        for j in range(args.num_op):
            #wave
            space['wav_policy_%d_%d'%(i,j)]=hp.choice('wav_policy_%d_%d'%(i,j),list(range(0,len(wav_ops))))
            space['wav_prob_%d_%d'%(i,j)]=hp.uniform('wav_prob_%d_%d'%(i,j),0.0,1.0)
            #spec
            # space['spec_policy_%d_%d' % (i, j)] = hp.choice('spec_policy_%d_%d' % (i, j), list(range(0, len(spec_ops))))
            # space['spec_prob_%d_%d' % (i, j)] = hp.uniform('spec_prob_%d_ %d' % (i, j), 0.0, 1.0)
            # space['spec_level_%d_%d' % (i, j)] = hp.uniform('spec_level_%d_ %d' % (i, j), 0.0, 1.0)
            #cv
            space['cv_policy_%d_%d' % (i, j)] = hp.choice('cv_policy_%d_%d' % (i, j), list(range(0, len(cv_ops))))
            space['cv_prob_%d_%d' % (i, j)] = hp.uniform('cv_prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['cv_level_%d_%d' % (i, j)] = hp.uniform('cv_level_%d_ %d' % (i, j), 0.0, 1.0)

          # for j in range(args.num_op):
          #   space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
          #   space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
          #   space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)

    ### path 这里把finetuned好的模型传在了这里
    path=C.get()['model']['finetuned_mdl_path']
    final_policy_set = []
    final_policy_set_map_threshold=[]
    total_computation = 0
    reward_attr = 'mAP'      # top1_valid or minus_loss
    for _ in range(1):  # run multiple times.
        for cv_fold in range(cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (C.get()['dataset'], C.get()['model']['type'], cv_fold, args.cv_ratio)
            print(name)
            register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_c), augs, rpt))
            algo = HyperOptSearch(space, max_concurrent=4*20, reward_attr=reward_attr)

            exp_config = {
                name: {
                    'run': name,
                    'num_samples': 4 if args.smoke_test else args.num_search,
                    'resources_per_trial': {'gpu': 1},
                    'stop': {'training_iteration': args.num_policy},
                    'config': {
                        'dataroot': args.dataroot, 'save_path': path,
                        'cv_ratio_test': args.cv_ratio, 'cv_fold': cv_fold,
                        'num_op': args.num_op, 'num_policy': args.num_policy
                    },
                }
            }
            results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True, resume=args.resume, raise_on_failed_trial=False)
            print()
            results = [x for x in results if x.last_result is not None]
            results = sorted(results, key=lambda x: x.last_result[reward_attr], reverse=True)

            # calculate computation usage
            for result in results:
                total_computation += result.last_result['elapsed_time']

            for result in results[:num_result_per_cv]:
                final_policy = policy_decoder(result.config, args.num_policy, args.num_op)
                # logger.info('loss=%.12f top1_valid=%.4f %s' % (result.last_result['minus_loss'], result.last_result['top1_valid'], final_policy))
                logger.info('loss=%.12f mAP=%.4f %s' % (result.last_result['minus_loss'], result.last_result['mAP'], final_policy))
                final_policy = remove_deplicates(final_policy)
                final_policy_set.extend(final_policy)

            for result in results[:num_result_per_cv]:
                final_policy = policy_decoder(result.config, args.num_policy, args.num_op)
                # logger.info('loss=%.12f top1_valid=%.4f %s' % (result.last_result['minus_loss'], result.last_result['top1_valid'], final_policy))
                # logger.info('loss=%.12f mAP=%.4f %s' % (result.last_result['minus_loss'], result.last_result['mAP'], final_policy))
                # print(final_policy)
                if(result.last_result['mAP']>=0.38):
                    final_policy = remove_deplicates(final_policy)
                    final_policy_set_map_threshold.extend(final_policy)
                else:
                    break

    logger.info(json.dumps(final_policy_set))

    final_policy_path = 'final_policy_ratio-'+str(args.cv_ratio)+'_search-'+str(args.num_search)+time.ctime()+'.json'
    final_policy_path_map = 'final_policy_with_map_sota-'+str(args.cv_ratio)+'_search-'+str(args.num_search)+time.ctime()+'.json'

    with open(final_policy_path, 'w') as write_f:
        write_f.write(json.dumps(final_policy_set, ensure_ascii=False))
    with open(final_policy_path_map,'w') as write_f:
        write_f.write(json.dumps(final_policy_set_map_threshold,ensure_ascii=False))
    

    logger.info('final_policy_of_top10=%d' % len(final_policy_set))
    logger.info('final_policy_of_sota=%d' % len(final_policy_set_map_threshold))
    logger.info('processed in %.4f secs, gpu hours=%.4f' % (w.pause('search'), total_computation / 3600.))
###         delete stage3           ###
    logger.info(w)
