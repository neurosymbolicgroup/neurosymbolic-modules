package org.genesys.models;

import java.util.List;

/**
 * Created by yufeng on 7/7/17.
 */
public class Component {

    @Override
    public String toString() {
        return "Component{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", constraint=" + constraint +
                '}';
    }

    public Integer getId() {
        return id;
    }

    public boolean isHigh() { return high; }

    public String getName() {
        return name;
    }

    public String getType() {
        return type;
    }

    public String getBit() { return bit; }

    public List<String> getConstraint() {
        return constraint;
    }

    private Integer id;

    private String name;

    /* Return type of the component. */
    private String type;

    /* Is this a function? */
    private boolean high;

    private String bit;

    private List<String> constraint;

}
