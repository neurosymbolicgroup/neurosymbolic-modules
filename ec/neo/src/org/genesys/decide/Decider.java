package org.genesys.decide;

import java.util.Map;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public interface Decider {

    /* Given a set of candidate production rules and a 'trail' containing the previous productions rules,
     * choose one of the candidate production rules and return it.
     */
    public String decide(List<String> trail, List<String> candidates);

    public String decideSketch(List<String> trail, List<String> candidates, int child);


}
