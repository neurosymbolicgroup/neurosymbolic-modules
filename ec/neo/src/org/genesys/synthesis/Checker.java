package org.genesys.synthesis;

import org.genesys.models.Node;

/**
 * S will be instantiated to the actual problem.
 * Created by yufeng on 5/28/17.
 */
public interface Checker<S, C> {
    boolean check(S specification, Node node);

    boolean check(S specification, Node node, Node curr);

    C learnCore();
}