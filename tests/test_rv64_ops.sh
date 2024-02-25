#!/bin/bash -

if [ ! "${QEMU_CMD}" ]; then
    echo '$QEMU_CMD is required.'
    exit -1
fi

test_models=(
"${QEMU_CMD} ./tests/test_op_absval"
"${QEMU_CMD} ./tests/test_op_add_n"
"${QEMU_CMD} ./tests/test_op_argmax"
"${QEMU_CMD} ./tests/test_op_argmin"
"${QEMU_CMD} ./tests/test_op_batchnorm"
"${QEMU_CMD} ./tests/test_op_batchtospacend"
# "${QEMU_CMD} ./tests/test_op_broadmul"
"${QEMU_CMD} ./tests/test_op_bias"
"${QEMU_CMD} ./tests/test_op_cast"
"${QEMU_CMD} ./tests/test_op_ceil"
"${QEMU_CMD} ./tests/test_op_clip"
"${QEMU_CMD} ./tests/test_op_comparison"
"${QEMU_CMD} ./tests/test_op_conv"
)

for (( i = 0 ; i < ${#test_models[@]} ; i++ ))
do
    echo ${test_models[$i]}
    echo ${test_models[$i]} | xargs -i sh -c "{}"

    if [ "$?" != 0 ]; then
        echo "failed"
        exit 1
    fi
done
