#!/bin/bash -e

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# グループを作成する
if [ x"$GROUP_ID" != x"0" ]; then
    groupadd -g $GROUP_ID $USER_NAME
fi

# ユーザを作成する
if [ x"$USER_ID" != x"0" ]; then
    useradd -d /home/$USER_NAME -m -s /bin/bash -u $USER_ID -g $GROUP_ID $USER_NAME
fi

# パーミッションを元に戻す
sudo chmod u-s /usr/sbin/useradd
sudo chmod u-s /usr/sbin/groupadd

# jupyter environment
echo ""
echo "Jupyter Settings =========================================="
mkdir ${HOME}/.jupyter
jupyter notebook --generate-config
{ \
    echo "c.NotebookApp.ip = '*'"; \
    echo "c.NotebookApp.password = u'sha1:0c1f8558e4f6:7938a25e46bdc554ff3ec5d574dbba543f1657f5'"; \  # (qwerty)
    # echo "c.NotebookApp.password = u'sha1:41418cb8a813:16be7213aa99abf406bdebcb11d08d0750c06d08'"; \  # (own passwd)
    echo "c.NotebookApp.open_browser = False"; \
    echo "c.NotebookApp.port = 8888"; \
} | tee -a ${HOME}/.jupyter/jupyter_notebook_config.py
echo " Jupyter Settings; done."
echo "==========================================================="
echo "To start jupyter server, please execute the following command:"
echo "jupyter lab --ip=0.0.0.0 --port=8888 --allow-root > /dev/null 2>&1 &"
echo ""

exec $@
