package org.genesys.synthesis;

import com.microsoft.z3.BoolExpr;
import org.genesys.models.Node;

/**
 * Created by yufeng on 5/28/17.
 */
public class Deductor<S, BoolExpr> implements Checker<S, BoolExpr> {
    @Override
    public boolean check(S specification, Node node) {
        return true;
    }

    @Override
    public boolean check(S specification, Node node, Node curr) {
        return false;
    }

    @Override
    public BoolExpr learnCore() {
        return null;
    }
}
