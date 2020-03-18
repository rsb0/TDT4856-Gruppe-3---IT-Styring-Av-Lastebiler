import React from 'react';
import { createStackNavigator } from 'react-navigation-stack';
import { createBottomTabNavigator } from 'react-navigation-tabs';

import TabBarIcon from '../components/TabBarIcon';
import PricesScreen from '../screens/PricesScreen';
import CameraScreen from "../screens/CameraScreen";
import MapScreen from "../screens/MapScreen";

const PriceListStack = createStackNavigator({
    Prices: PricesScreen,
    }
);

PricesScreen.navigationOptions = {
    header: null,
};

PriceListStack.navigationOptions = {
    tabBarLabel: "Prices",
    tabBarIcon: ({ focused }) => (
        <TabBarIcon focused={focused} name={Platform.OS === 'ios' ? 'ios-list' : 'md-list'}/>
    )
};

PriceListStack.path = "";

const CameraStack = createStackNavigator({
        Camera: CameraScreen,
    }
);

CameraScreen.navigationOptions = {
    header: null,
};

CameraStack.navigationOptions = {
    tabBarLabel: "Camera",
    tabBarIcon: ({ focused }) => (
        <TabBarIcon focused={focused} name={Platform.OS === 'ios' ? 'ios-camera' : 'md-camera'}/>
    )
};

CameraStack.path = "";

const MapStack = createStackNavigator({
        Map: MapScreen,
    }
);

MapScreen.navigationOptions = {
    header: null,
};

MapStack.navigationOptions = {
    tabBarLabel: "Map",
    tabBarIcon: ({ focused }) => (
        <TabBarIcon focused={focused} name={Platform.OS === 'ios' ? 'ios-map' : 'md-map'}/>
    )
};

MapStack.path = "";

const tabNavigator = createBottomTabNavigator({
  PriceListStack,
  MapStack,
  CameraStack,
});

tabNavigator.path = '';

export default tabNavigator;
