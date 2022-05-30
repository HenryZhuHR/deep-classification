# scp -r -P 23 zhr@202.114.107.172:~/project/classification/* ./

rsync -r -e 'ssh -p 23' -vv \
    --exclude={'*.pyc','__pycache__','.vscode'} \
    --link-dest ./ \
    zhr@202.114.107.172:~/project/classification/* ./

