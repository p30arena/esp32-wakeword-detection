#define SPG_N_BLOCKS 32
#define SPG_BLOCK_SIZE 512

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
static double o_11[SPG_BLOCK_SIZE] = {0};
static double o_12[SPG_BLOCK_SIZE] = {0};
static double o_13[SPG_BLOCK_SIZE] = {0};
static double o_14[SPG_BLOCK_SIZE] = {0};

static double **xyz;

double **getSPGBuffer()
{
  xyz = new double *[SPG_N_BLOCKS];

  for (int i = 0; i < 18; i++)
  {
    xyz[i] = new double[SPG_BLOCK_SIZE];
  }

  xyz[18] = o_1;
  xyz[19] = o_2;
  xyz[20] = o_3;
  xyz[21] = o_4;
  xyz[22] = o_5;
  xyz[23] = o_6;
  xyz[24] = o_7;
  xyz[25] = o_8;
  xyz[26] = o_9;
  xyz[27] = o_10;
  xyz[28] = o_11;
  xyz[29] = o_12;
  xyz[30] = o_13;
  xyz[31] = o_14;

  return xyz;
}