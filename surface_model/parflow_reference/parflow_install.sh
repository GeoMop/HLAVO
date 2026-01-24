set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo apt install tcl tcl-dev

cd $SCRIPT_DIR
#wget -O parflow_source.tar.gz https://github.com/parflow/parflow/archive/refs/tags/v3.13.0.tar.gz 
tar -zxvf parflow_source.tar.gz
PARFLOW_SRC=`ls -d parflow-*`
export PARFLOW_DIR=$SCRIPT_DIR/parflow
mkdir -f $PARFLOW_DIR

mkdir build
cd build
cmake ../$PARFLOW_SRC -DCMAKE_INSTALL_PREFIX=${PARFLOW_DIR} -DPARFLOW_HAVE_CLM=ON
make 
make install

# update .bashrc

echo "export PARFLOW_DIR=$PARFLOW_DIR" >> ~/.bashrc

echo "verify ~/.bashrc, here is its tail:"
tail ~/.bashrc
