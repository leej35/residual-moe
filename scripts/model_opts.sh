
if [ "${model_type}" == "MyLSTM" ] || [ "${model_type}" == "MyGRU" ] || \
    [ "${model_type}" == "GRU" ] || [ "${model_type}" == "LSTM" ] || \
    [ "${model_type}" == "MyLayerLSTM" ]
then
    model_opt=" --rnn-type ${model_type}"

elif [ "${model_type}" == "RETAIN" ] 
then
    model_opt=" --rnn-type retain"

elif [ "${model_type}" == "CNN" ] 
then
    model_opt=" --rnn-type CNN"

else
    model_opt=" --baseline-type ${model_type}"
fi


