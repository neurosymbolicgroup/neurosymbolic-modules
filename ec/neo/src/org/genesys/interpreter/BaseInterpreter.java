package org.genesys.interpreter;

import org.genesys.models.Node;
import org.genesys.models.Pair;
import org.genesys.type.Maybe;

import java.util.*;

/**
 * Created by yufeng on 5/30/17.
 */
public class BaseInterpreter implements Interpreter<Node, Object> {

    public final Map<String, Executor> executors = new HashMap<>();

    private final boolean cacheOn_ = false;

    private final Map<Pair<Node, Object>, Maybe> cache_ = new HashMap<>();

    private final int bad_num = 256;

    @Override
    public Maybe<Object> execute(Node node, Object input) {
        List<Object> arglist = new ArrayList<>();

        for (Node child : node.children) {
            Maybe<Object> object = this.execute(child, input);
            if (!object.has()) {
                return object;
            } else {
                Object o = object.get();
                if (o instanceof Integer) {
                    int inVal = (int) o;
                    if (bad_num == inVal) {
                        return new Maybe<>(bad_num);
                    }
                }
            }
            arglist.add(object.get());
        }

        if (!this.executors.containsKey(node.function)) {
            throw new UnsupportedOperationException("Invalid argument." + node.function);
        }
        assert arglist.size() == node.children.size();

        if (cacheOn_) {
            Pair<Node, Object> key = new Pair<>(node, input);
            Maybe m;
            if (cache_.containsKey(key)) {
                m = cache_.get(key);
            } else {
                m = this.executors.get(node.function).execute(arglist, input);
                cache_.put(key, m);
            }
            return m;
        } else {
            return this.executors.get(node.function).execute(arglist, input);
        }
    }

    @Override
    public Set<String> getExeKeys() {
        throw new UnsupportedOperationException("Unsupported interpreter: Default.");
    }
}
