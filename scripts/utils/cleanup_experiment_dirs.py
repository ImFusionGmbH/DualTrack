from argparse import ArgumentParser
import shutil
import torch
import os


def main():
    p = ArgumentParser() 
    p.add_argument('root')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    for path in find_dirs(args.root): 
        print('===============')
        print(path)

        if 'WANDB_URL' in os.listdir(path): 
            wandb_url = open(os.path.join(path, "WANDB_URL")).read()
            print(wandb_url)
            entity = wandb_url.split('/')[-3]
            project = wandb_url.split('/')[-2]
            id = wandb_url.split('/')[-1]
            
            import wandb
            api = wandb.Api()
            run = api.run("/".join([entity, project, id]))
            print(run.summary)
            print(run.tags)
            print(run.summary.get('epoch'))

            if not run.summary.get('epoch'): 
                print(f"No epoch completed. Deleting {path}...")
                if not args.dry_run: 
                    shutil.rmtree(path)


def find_dirs(root): 
    for path in os.listdir(root): 
        path = os.path.join(root, path)
        print(path)

        if not os.path.isdir(path): 
            continue 

        if 'checkpoint' in os.listdir(path): 
            yield path 

        else: 
            yield from find_dirs(path)


if __name__ == "__main__": 
    main()