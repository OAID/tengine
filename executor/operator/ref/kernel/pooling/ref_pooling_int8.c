
<<<<<<< HEAD
static inline int calc_sum_int8(const int8_t* input, int layout, int c, int h, int w, int cur_ch, int start_h, int start_w, int end_h, int end_w)
{
    int sum = 0;
    for(int i=start_h;i<end_h;i++)
        for(int j=start_w;j<end_w;j++)
        {
            if(layout == 0)
                sum += input[cur_ch*h*w + i*w + j];
            else
                sum += input[i*w*c + j* c + cur_ch];

=======
static inline int calc_sum_int8(const int8_t* input, int layout, int c, int h, int w, int cur_ch, int start_h,
                                int start_w, int end_h, int end_w)
{
    int sum = 0;
    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                sum += input[cur_ch * h * w + i * w + j];
            else
                sum += input[i * w * c + j * c + cur_ch];
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
        }

    return sum;
}

<<<<<<< HEAD
static inline int8_t calc_max_int8(const int8_t* input, int layout, int c, int h, int w, int cur_ch, int start_h, int start_w, int end_h, int end_w)
{
    int8_t max = 0;
    if(layout == 0)
        max = input[cur_ch*h*w + start_h*w + start_w];
    else
        max = input[start_h*w*c + start_w*c + cur_ch];
    
    int8_t tmp = 0;
    for(int i=start_h;i<end_h;i++)
        for(int j=start_w;j<end_w;j++)
        {
            if(layout == 0)
                tmp = input[cur_ch*h*w + i*w + j];
            else
                tmp = input[i*w*c + j* c + cur_ch];

            max = max>tmp ? max : tmp;

=======
static inline int8_t calc_max_int8(const int8_t* input, int layout, int c, int h, int w, int cur_ch, int start_h,
                                   int start_w, int end_h, int end_w)
{
    int8_t max = 0;
    if(layout == 0)
        max = input[cur_ch * h * w + start_h * w + start_w];
    else
        max = input[start_h * w * c + start_w * c + cur_ch];

    int8_t tmp = 0;
    for(int i = start_h; i < end_h; i++)
        for(int j = start_w; j < end_w; j++)
        {
            if(layout == 0)
                tmp = input[cur_ch * h * w + i * w + j];
            else
                tmp = input[i * w * c + j * c + cur_ch];

            max = max > tmp ? max : tmp;
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
        }

    return max;
}

<<<<<<< HEAD
static int ref_pooling_int8(const int8_t* input, int8_t * output, struct op_data* param)
{
    int input_chw = param->channel * param->input[0]*param->input[1];
    int output_chw = param->channel * param->output[0]*param->output[1];
    
    for(int n = 0; n < param->batch; n++)
    {
        const int8_t* input_cur = input + n*input_chw;
=======
static int ref_pooling_int8(const int8_t* input, int8_t* output, struct op_data* param)
{
    int input_chw = param->channel * param->input[0] * param->input[1];
    int output_chw = param->channel * param->output[0] * param->output[1];

    for(int n = 0; n < param->batch; n++)
    {
        const int8_t* input_cur = input + n * input_chw;
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
        for(int c = 0; c < param->channel; c++)
        {
            for(int ph = 0; ph < param->output[0]; ph++)
            {
                for(int pw = 0; pw < param->output[1]; pw++)
                {
                    int pool_size = 1;
                    int offset = 0;
                    int h_start = ph * param->strides[0] - param->pads[0];
                    int h_end = h_start + param->kernels[0];
<<<<<<< HEAD
                    if( h_end > param->input[0] + param->pads[0])
                        h_end = param->input[0] + param->pads[0];
                    int w_start = pw * param->strides[1] - param->pads[1];
                    int w_end = w_start + param->kernels[1];
                    if( w_end > param->input[1] + param->pads[1])
=======
                    if(h_end > param->input[0] + param->pads[0])
                        h_end = param->input[0] + param->pads[0];
                    int w_start = pw * param->strides[1] - param->pads[1];
                    int w_end = w_start + param->kernels[1];
                    if(w_end > param->input[1] + param->pads[1])
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
                        w_end = param->input[1] + param->pads[1];

                    if(param->caffe_flavor)
                        pool_size = (h_end - h_start) * (w_end - w_start);

                    h_start = h_start > 0 ? h_start : 0;
                    w_start = w_start > 0 ? w_start : 0;
                    h_end = h_end < param->input[0] ? h_end : param->input[0];
                    w_end = w_end < param->input[1] ? w_end : param->input[1];
<<<<<<< HEAD
                    //printf("w: %d,%d ,h: %d,%d\n",w_start,w_end,h_start,h_end);

                    if(!param->caffe_flavor)
                        pool_size = (h_end - h_start) * (w_end - w_start);
                    if(param->layout == 0)     //nchw
                        offset = n*output_chw + c*param->output[0]*param->output[1]
                                + ph*param->output[1] + pw;
                    else
                        offset =n*output_chw + ph*param->output[1]*param->channel
                                + pw*param->channel + c;
                    
                    if(param->method == 0)
                    {
                        int8_t max = calc_max_int8(input_cur,param->layout,param->channel,param->input[0],param->input[1],
                                            c,h_start,w_start,h_end,w_end);
                        output[offset] = max;
                    }
                    else if( param->method == 1)
                    {
                        int sum = calc_sum_int8(input_cur,param->layout,param->channel,param->input[0],param->input[1],
                                            c,h_start,w_start,h_end,w_end);
                        output[offset] = (int8_t)round(sum/pool_size);
=======
                    // printf("w: %d,%d ,h: %d,%d\n",w_start,w_end,h_start,h_end);

                    if(!param->caffe_flavor)
                        pool_size = (h_end - h_start) * (w_end - w_start);
                    if(param->layout == 0)    // nchw
                        offset = n * output_chw + c * param->output[0] * param->output[1] + ph * param->output[1] + pw;
                    else
                        offset = n * output_chw + ph * param->output[1] * param->channel + pw * param->channel + c;

                    if(param->method == 0)
                    {
                        int8_t max = calc_max_int8(input_cur, param->layout, param->channel, param->input[0],
                                                   param->input[1], c, h_start, w_start, h_end, w_end);
                        output[offset] = max;
                    }
                    else if(param->method == 1)
                    {
                        int sum = calc_sum_int8(input_cur, param->layout, param->channel, param->input[0],
                                                param->input[1], c, h_start, w_start, h_end, w_end);
                        output[offset] = ( int8_t )round(sum / pool_size);
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
                    }
                    else
                        return -1;
                }
            }
        }
    }
    return 0;
}
