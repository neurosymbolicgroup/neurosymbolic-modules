package org.genesys.language;

/**
 * Created by yufeng on 5/26/17.
 */
public class Symbol {

    boolean terminal = false;

    String name;

    public Symbol(String val) {
        name = val;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public boolean isTerminal() {
        return terminal;
    }

    @Override
    public String toString() {
        return name;
    }
}
