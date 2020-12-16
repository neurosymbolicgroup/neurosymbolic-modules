package org.genesys.models;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.genesys.language.Production;

/**
 * Created by yufeng on 5/26/17.
 */
public class Node {

    private boolean pe_ = false;

    private Object symbol;

    public int id = -1;

    public int component = -1;

    public int level = -1;

    public String function;

    public List<Node> children = new ArrayList<>();

    public HashMap<Production, Boolean> activeDomain = new HashMap<>();

    public List<Production> domain = new ArrayList<>();

    public Production decision = null;

    public Node() {
        this.function = "";
        this.children = new ArrayList<>();
        this.domain = new ArrayList<>();
        this.activeDomain = new HashMap<>();
    }

    public Node(String function, List<Node> children) {
        this.function = function;
        this.children = children;
    }

    public Node(String function, List<Node> children, List<Production> productions) {
        this.function = function;
        this.children = children;

        for (Production p : productions) {
            activeDomain.put(p, true);
        }
        domain = productions;
    }

    public void setDomain(List<Production> productions) {
        for (Production p : productions) {
            activeDomain.put(p, true);
        }
        domain = productions;
    }

    public boolean isConcrete() {
        return pe_;
    }

    public void setConcrete(boolean value) {
        pe_ = value;
    }

    public void addChild(Node node) {
        children.add(node);
    }

    public Node(String function) {
        this.function = function;
    }

    public Object getSymbol() {
        return symbol;
    }

    public void setSymbol(Object symbol) {
        this.symbol = symbol;
    }

    @Override
    public boolean equals(Object o) {
        assert o instanceof Node;
        Node other = (Node) o;
        //if (!function.equals(other.function)) return false;
        if (component != other.component) return false;
        if (children.size() != other.children.size()) return false;
        return children.equals(other.children);
    }

    @Override
    public int hashCode() {
        int result = 17;
        //result = 31 * result + function.hashCode();
        result = 31 * result + component;
        for (Node child : children) {
            result = 31 * result + child.hashCode();
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (this.children.size() > 0) {
            sb.append("(");
        }
        sb.append(this.function).append(" ");
        for (Node child : this.children) {
//            sb.append("[id=" + child.id + "]");
            sb.append(child.toString()).append(" ");
        }
        sb.deleteCharAt(sb.length() - 1);
        if (this.children.size() > 0) {
            sb.append(")");
        }
        return sb.toString();
    }

}
