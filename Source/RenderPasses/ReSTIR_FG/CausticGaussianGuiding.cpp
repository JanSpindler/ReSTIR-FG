#include "CausticGaussianGuiding.h"

struct CausticGaussianReservoir
{
    float3 pos;
    float thp;
    float weightSum;
    float confidence;

    constexpr CausticGaussianReservoir() : pos(0.0f), thp(0.0f), weightSum(0.0f), confidence(0.0f) {}
};

struct CausticSource
{
    float3 srcPos; // Position of the caustic castic diffuse surface
    float3 targetPos; // Position of the first vertex after the source in the caustic chain
};

void CausticGaussianGuiding::PrepareBuffers(RenderContext* pRenderContext, const uint causticBufSize)
{
    for (uint idx = 0; idx < 2; ++idx)
    {
        const std::vector<CausticGaussianReservoir> emptyReservoirs(m_HashGridSize, CausticGaussianReservoir());
        if (!m_ReservoirHashGrid[idx])
        {
            m_ReservoirHashGrid[idx] = Buffer::createStructured(
                m_Device, sizeof(CausticGaussianReservoir), m_HashGridSize,
                ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, emptyReservoirs.data()
            );
        }
    }

    if (!m_HashGridCounter)
    {
        m_HashGridCounter = Buffer::create(m_Device, sizeof(uint) * m_HashGridSize);
    }

    if (!m_CausticSourceMap or m_CausticSourceMap->getElementCount() < causticBufSize)
    {
        m_CausticSourceMap = Buffer::createStructured(m_Device, sizeof(CausticSource), causticBufSize);
    }
}

bool CausticGaussianGuiding::RenderUI(Gui::Widgets& widget)
{
    bool changed = false;

    if (auto group = widget.group("CausticGaussianGuiding"))
    {
        changed |= group.checkbox("Caustic Gaussian Guiding", m_Active);

        if (group.var("Hash Grid Size", m_HashGridSize, 1u))
        {
            m_ReservoirHashGrid[0].reset();
            m_ReservoirHashGrid[1].reset();
            m_HashGridCounter.reset();
            changed = true;
        }

        changed |= group.var("Hash Scaling Factor", m_HashScalingFactor, 1.0f, 1e6f);

        changed |= group.var("Gauss Sampling Prob", m_GaussSamplingProb, 0.0f, 1.0f);
    }

    return changed;
}

void CausticGaussianGuiding::Run(RenderContext* pRenderContext)
{
    // TODO: Spatial resampling in hash grid
}

void CausticGaussianGuiding::ClearHashGridCounter(RenderContext* pRenderContext) const
{
    pRenderContext->clearUAV(m_HashGridCounter->getUAV().get(), uint4(0, 0, 0, 0));
}

void CausticGaussianGuiding::SetGeneratePhotonsVars(const ShaderVar& vars)
{
    vars["CausticGaussianGuiding"]["gCausticHashGridSize"] = m_HashGridSize;
    vars["CausticGaussianGuiding"]["gCausticHashScalingFactor"] = m_HashScalingFactor;
    vars["CausticGaussianGuiding"]["gCausticGaussianSamplingProb"] = m_GaussSamplingProb;

    vars["gCausticGaussianReservoirs"] = m_ReservoirHashGrid[m_CurrentBufIdx];
    vars["gCausticSources"] = m_CausticSourceMap;

    ++m_CurrentBufIdx;
    m_CurrentBufIdx %= 2;
}

void CausticGaussianGuiding::SetCollectPhotonsVars(const ShaderVar& vars)
{
    vars["CausticGaussianGuiding"]["gCausticHashGridSize"] = m_HashGridSize;
    vars["CausticGaussianGuiding"]["gCausticHashScalingFactor"] = m_HashScalingFactor;

    vars["gCausticGaussianReservoirs"] = m_ReservoirHashGrid[m_CurrentBufIdx];
    vars["gCausticSources"] = m_CausticSourceMap;
}
