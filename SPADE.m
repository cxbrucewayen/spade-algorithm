% SPADE: Sample-based Pre-Assessment Differential Evolution
% 基于采样预评估的差分进化算法
function [BestSol, Convergence_curve] = SPADE(SearchAgents_no, MaxFEs, lb, ub, dim, fobj)
    nPop = SearchAgents_no;
    MaxIt = MaxFEs;
    lb = ones(1, dim) .* lb;
    ub = ones(1, dim) .* ub;
    FEs = 0;
    empty_individual.Position = [];
    empty_individual.Cost = [];
    BestSol.Cost = inf;
    pop = repmat(empty_individual, nPop, 1);
    for i = 1:nPop
        pop(i).Position = init_individual(lb, ub, dim, 1);
        pop(i).Cost = fobj(pop(i).Position);
        FEs = FEs + 1;
        if pop(i).Cost < BestSol.Cost
            BestSol = pop(i);
        end
    end
    Convergence_curve = [];
    it = 1;
    Diag = norm(ub - lb);
    while FEs < MaxIt
        costs = arrayfun(@(s) s.Cost, pop);
        minCost = min(costs);
        maxCost = max(costs);
        nElite = max(1, round(0.1 * nPop));
        [~, sortIdx] = sort(costs);
        eliteIdx = sortIdx(1:nElite);
        for i = 1:nPop
            x = pop(i).Position;
            A = randperm(nPop);
            A(A == i) = [];
            r1 = A(1);
            r2 = A(2);
            r3 = A(3);
            pbest = pop(eliteIdx(randi(nElite))).Position;
            F1 = 0.5 + 0.5 * rand;
            CR1 = 0.9;
            v1 = x + F1 * (pbest - x) + F1 * (pop(r1).Position - pop(r2).Position);
            u1 = x;
            j0 = randi(dim);
            for j = 1:dim
                if j == j0 || rand <= CR1
                    u1(j) = v1(j);
                end
            end
            F2 = 0.8 + 0.2 * rand;
            CR2 = 0.2;
            v2 = pop(r1).Position + F2 * (pop(r2).Position - pop(r3).Position);
            u2 = x;
            j0 = randi(dim);
            for j = 1:dim
                if j == j0 || rand <= CR2
                    u2(j) = v2(j);
                end
            end
            F3 = 0.5 + 0.5 * rand;
            u3 = x + rand * (pop(r1).Position - x) + F3 * (pop(r2).Position - pop(r3).Position);
            candidates = [u1; u2; u3];
            for k = 1:3
                for j = 1:dim
                    if candidates(k, j) < lb(j)
                        candidates(k, j) = (x(j) + lb(j)) / 2;
                    elseif candidates(k, j) > ub(j)
                        candidates(k, j) = (x(j) + ub(j)) / 2;
                    end
                end
            end
            m = min(5, nPop);
            refIdx = randperm(nPop, m);
            scores = zeros(1, 3);
            for k = 1:3
                minDist = inf;
                nearestCost = inf;
                for q = 1:m
                    d = norm(candidates(k, :) - pop(refIdx(q)).Position);
                    if d < minDist
                        minDist = d;
                        nearestCost = pop(refIdx(q)).Cost;
                    end
                end
                scores(k) = (nearestCost - minCost) / (maxCost - minCost + 1e-15) - 0.1 * (minDist / (Diag + 1e-15));
            end
            [~, bestK] = min(scores);
            NewSol.Position = candidates(bestK, :);
            NewSol.Cost = fobj(NewSol.Position);
            FEs = FEs + 1;
            if NewSol.Cost < pop(i).Cost
                pop(i) = NewSol;
                if pop(i).Cost < BestSol.Cost
                    BestSol = pop(i);
                end
            end
        end
        Convergence_curve(it) = BestSol.Cost;
        it = it + 1;
    end
end

function x = init_individual(xlb, xub, dim, sizepop)
    xRange = repmat((xub - xlb), [sizepop, 1]);
    xLower = repmat(xlb, [sizepop, 1]);
    x = rand(sizepop, dim) .* xRange + xLower;
end