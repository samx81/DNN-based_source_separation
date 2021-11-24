if [ -z $1 ]; then
	echo "Parameter not supplied";
	exit 1;
fi

alter_exp=""
if [ -z $2 ]; then
	echo "Using defalut exp dir."
else
	alter_exp=$2
fi

tag=$1
# enhance_dir=$2

exp_test_dir=${PWD}/exp${alter_exp}/${tag}/test/
echo "${exp_test_dir}, ${enhance_dir}"
cp local/sep.scp.bak "${exp_test_dir}/sep.scp"
if [ ! -z $2 ]; then
	alter_exp_sed="s/exp/exp${alter_exp}/g"
	echo "Alter exp: ${alter_exp_sed}"
	sed -i $alter_exp_sed ${exp_test_dir}/sep.scp
fi 
sed_str="s/replaceme/${tag}/g"
echo $sed_str
sed -i $sed_str ${exp_test_dir}/sep.scp
cd tenet
./inf_galr.sh $exp_test_dir

bat ${exp_test_dir}/pesq-wb
