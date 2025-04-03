[numthreads(32, 1, 1)]
void main()
{
    // Get light index
    const uint gaussianIdx = index.x;
    if (gaussianIdx >= gTotalGaussianCount)
    {
        return;
    }
}
