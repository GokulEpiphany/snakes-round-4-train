
import time
from os.path import join as pjoin
import numpy as np
import torch
import torchvision as tv
import bit_common
import bit_hyperrule

import big_transfer.models as models
import big_transfer.lbtoolbox as lb
from dataset import SnakeDataset
import pandas as pd

def topk(output,target,ks=(1,)):
  _,pred = output.topk(max(ks),1,True,True)
  pred = pred.t()
  correct = pred.eq(target.view(1,-1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def mkval(args):
  precrop,crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  valid_tx = tv.transforms.Compose([
    tv.transforms.Resize((crop,crop)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

  path = args.datadir
  validate_csv_file = pjoin(path,'metadata','validate_labels.csv')

  valid_set = SnakeDataset(path,is_train=False,transform=valid_tx,target_transform=None,csv_file=validate_csv_file)

  valid_loader = torch.utils.data.DataLoader(
  valid_set,batch_size = args.batch_size,shuffle=False,num_workers=0,pin_memory=True,drop_last=False)

  return valid_set,valid_loader,valid_set.classes


def run_eval(model,data_loader,device,chrono,logger,classes):
  model.eval()
  logger.info("Running validation")
  logger.flush()

  all_c,all_top1,all_top5 = [],[],[]
  end = time.time()
  results = np.empty((0,783),float)
  filenames=[]
  for b,(x,y,name) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device,non_blocking=True)
      y = y.to(device,non_blocking=True)
      logger.info("In b {}".format(b))
      logits = model(x)
      softmax_op = torch.nn.Softmax(dim=1)
      probs = softmax_op(logits)
      data_to_save = probs.data.cpu().numpy()
      results = np.concatenate((results,data_to_save),axis=0)
      logger.info(len(name))
      filenames.append(np.squeeze(name))
      c = torch.nn.CrossEntropyLoss(reduction='none')(logits,y)
      top1,top5 = topk(logits,y,ks=(1,5))
      all_c.extend(c.cpu())
      all_top1.extend(top1.cpu())
      all_top5.extend(top5.cpu())

  logger.info(f"top1 {np.mean(all_top1):.2%},"
                f"top 5 {np.mean(all_top5):.2%}")

  #Create csv file
  logger.info("Length of index is {}".format(len(filenames)))
  logger.info(data_to_save.shape)
  logger.flush()
  #flatten name list
  flat_filenames = [y for x in filenames for y in x]
  df = pd.DataFrame(data=results,index=flat_filenames,columns=classes)
  df.index.name = 'hashed_id'
  df.to_csv('results_check.csv',index=True)
  return all_c,all_top1,all_top5



def main(args):
  print(f"Args save is {args.save}")
  logger = bit_common.setup_logger(args)
  torch.backends.cudnn.benchmark=True
  device = torch.device("cuda:0")
  assert torch.cuda.is_available()
  chrono = lb.Chrono()
  logger.info(f"Validating")
  valid_set,valid_loader,classes = mkval(args)
  logger.info(f"Loading model from {args.model}.npz")
  model = models.KNOWN_MODELS[args.model](head_size = len(valid_set.classes),zero_head=True)
  model = torch.nn.DataParallel(model)
  step = 0
  optim = torch.optim.SGD(model.parameters(),lr=0.003,momentum=0.9)
  savename = pjoin(args.logdir,args.name,"bit.pth.tar")
  checkpoint = torch.load(savename,map_location="cpu")
  model.load_state_dict(checkpoint["model"])
  model = model.to(device)
  run_eval(model,valid_loader,device,chrono,logger,classes)


if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir",required=True)
  parser.add_argument("--workers", type=int,default=0)
  parser.add_argument("--no-save",dest="save",action="store_false")
  parser.add_argument("--batch-size",type=int,default=8)
  main(parser.parse_args())



