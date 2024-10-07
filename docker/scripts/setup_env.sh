set -ex

set_bash_env() {
  if [ ! -f ${BASH_ENV} ];then
    touch ${BASH_ENV}
  fi
  # In the existing base images, as long as `ENV` is set, it will be enabled by `BASH_ENV`.
  if [ ! -f ${ENV} ];then
    touch ${ENV}
    (echo "test -f ${ENV} && source ${ENV}" && cat ${BASH_ENV}) > /tmp/shinit_f
    mv /tmp/shinit_f ${BASH_ENV}
  fi
}

set_bash_env
