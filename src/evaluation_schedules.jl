"""
    Distributed schedule of when to increase evaluation
    resources and then compensate for the increase later.
"""
function evaluation_schedule(P, k, k_max, m, m_elite; p_UNUSED=0.2)
    N_max = k_max*m
    m_sched = floor(pdf(P, k)*N_max)
    if k == k_max # account for left-overs and over counting
        subtotal = sum(floor(pdf(P, i)*N_max) for i in 1:(k_max-1))
        m_sched = min(N_max - subtotal, N_max - m_sched)
    end
    mₑ = Int(m_sched)
    m_elite = min(m_elite, mₑ) # Clamp.
    return (mₑ, m_elite)
end


"""
    Schedule of when to increase evaluation resources and then
    compensate for the increase later.
"""
function evaluation_schedule_manual(k, m)
    if k == 1
        m+=10
    elseif k == 2
        m+=5
    elseif k == 3
        m-=1
    elseif 4 ≤ k ≤ 10
        m-=2
    end
    return m
end