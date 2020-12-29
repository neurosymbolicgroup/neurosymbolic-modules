package org.genesys.decide;

import java.util.List;
import java.util.Random;

/**
 * Created by utcs on 8/26/17.
 */
public class FirstDecider implements Decider {

    @Override
    public String decide(List<String> trail, List<String> candidates) {
        return candidates.get(0);
    }

    @Override
    public String decideSketch(List<String> trail, List<String> candidates, int child) { return decide(trail, candidates); }

}
