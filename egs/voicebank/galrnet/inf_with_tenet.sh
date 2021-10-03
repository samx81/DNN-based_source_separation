if [ -z $1 ]; then
	echo "Parameter not supplied";
	exit 1;
fi

tag=$1
# enhance_dir=$2

exp_test_dir=${PWD}/exp/${tag}/test/
echo "${exp_test_dir}, ${enhance_dir}"
cp local/sep.scp.bak "${exp_test_dir}/sep.scp"
sed_str="s/replaceme/${tag}/g"
echo $sed_str
sed -i $sed_str ${exp_test_dir}/sep.scp
cd tenet
./inf_galr.sh $exp_test_dir
