#define SPG_N_BLOCKS 32
#define SPG_BLOCK_SIZE 512
#define SPG_IMG_W 32
#define SPG_IMG_SIZE (SPG_IMG_W * SPG_IMG_W)

static double o_1[SPG_BLOCK_SIZE] = {0};
static double o_2[SPG_BLOCK_SIZE] = {0};
static double o_3[SPG_BLOCK_SIZE] = {0};
static double o_4[SPG_BLOCK_SIZE] = {0};
static double o_5[SPG_BLOCK_SIZE] = {0};
static double o_6[SPG_BLOCK_SIZE] = {0};
static double o_7[SPG_BLOCK_SIZE] = {0};
static double o_8[SPG_BLOCK_SIZE] = {0};
static double o_9[SPG_BLOCK_SIZE] = {0};
static double o_10[SPG_BLOCK_SIZE] = {0};
// static double o_11[SPG_BLOCK_SIZE] = {0};
// static double o_12[SPG_BLOCK_SIZE] = {0};
// static double o_13[SPG_BLOCK_SIZE] = {0};
// static double o_14[SPG_BLOCK_SIZE] = {0};

double **spg_buffer;
double *spg_img_buffer;

void initSPGBuffer()
{
  spg_buffer = new double *[SPG_N_BLOCKS];

  for (int i = 0; i < 22; i++)
  {
    spg_buffer[i] = new double[SPG_BLOCK_SIZE];
  }

  spg_buffer[22] = o_1;
  spg_buffer[23] = o_2;
  spg_buffer[24] = o_3;
  spg_buffer[25] = o_4;
  spg_buffer[26] = o_5;
  spg_buffer[27] = o_6;
  spg_buffer[28] = o_7;
  spg_buffer[29] = o_8;
  spg_buffer[30] = o_9;
  spg_buffer[31] = o_10;
  // spg_buffer[28] = o_11;
  // spg_buffer[29] = o_12;
  // spg_buffer[30] = o_13;
  // spg_buffer[31] = o_14;

  spg_img_buffer = new double[SPG_IMG_SIZE];
}

void zeroSPGBuffer()
{
  for (int i = 0; i < SPG_N_BLOCKS; i++)
  {
    memset(spg_buffer[i], 0, sizeof(double) * SPG_BLOCK_SIZE);
  }

  memset(spg_img_buffer, 0, sizeof(double) * SPG_IMG_SIZE);
}