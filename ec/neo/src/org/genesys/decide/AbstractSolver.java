package org.genesys.decide;

import java.util.List;
import org.genesys.models.Pair;
import java.util.ArrayList;

/**
 * Created by yufeng on 5/31/17.
 */
public interface AbstractSolver<C, T> {

    T getModel(C core, boolean block);

    T getCoreModel(List<Pair<Integer, List<String>>> core, boolean block, boolean global);

    T getCoreModelSet(List<List<Pair<Integer, List<String>>>> core, boolean block, boolean global);

    boolean isPartial();

    void cacheAST(String program, boolean block);

    ArrayList<Double> getLearnStats();

}
